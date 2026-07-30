// Run a kernel from a binary archive, including one whose machine code has been edited.
//
// usage: agx_execute <archive.bin> <source.metal> <function> <inputs.f32> <outputs.f32> <threads>
//
// The archive is the authority. Metal is given the same MSL it was built from, because a
// pipeline descriptor needs a function object, but `failOnBinaryArchiveMiss` makes the archive
// the only permitted source of machine code: if the driver cannot find a matching entry it
// errors instead of quietly recompiling the source. That is what makes an edited archive
// observable rather than silently ignored — without the flag a patched kernel and an unpatched
// one produce the same answer and the experiment proves nothing.
//
// Inputs and outputs are raw float32 files so the caller can drive this from anywhere.

import Foundation
import Metal

let arguments = CommandLine.arguments
guard arguments.count == 7 else {
    FileHandle.standardError.write(
        "usage: agx_execute <archive.bin> <source.metal> <function> <in.f32> <out.f32> <threads>\n"
            .data(using: .utf8)!)
    exit(2)
}

guard let device = MTLCreateSystemDefaultDevice() else {
    FileHandle.standardError.write("no Metal device\n".data(using: .utf8)!)
    exit(1)
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write("error: \(message)\n".data(using: .utf8)!)
    exit(1)
}

do {
    let source = try String(contentsOfFile: arguments[2], encoding: .utf8)
    let library = try device.makeLibrary(source: source, options: MTLCompileOptions())
    guard let function = library.makeFunction(name: arguments[3]) else {
        fail("no function \(arguments[3])")
    }

    let descriptor = MTLBinaryArchiveDescriptor()
    descriptor.url = URL(fileURLWithPath: arguments[1])
    let archive = try device.makeBinaryArchive(descriptor: descriptor)

    let pipelineDescriptor = MTLComputePipelineDescriptor()
    pipelineDescriptor.computeFunction = function
    pipelineDescriptor.binaryArchives = [archive]
    let pipeline = try device.makeComputePipelineState(
        descriptor: pipelineDescriptor,
        options: .failOnBinaryArchiveMiss,
        reflection: nil)

    let inputData = try Data(contentsOf: URL(fileURLWithPath: arguments[4]))
    guard let threads = Int(arguments[6]), threads > 0 else { fail("threads must be positive") }
    let elements = max(inputData.count / 4, threads)

    guard
        let input = device.makeBuffer(length: elements * 4, options: .storageModeShared),
        let output = device.makeBuffer(length: elements * 4, options: .storageModeShared),
        let scalar = device.makeBuffer(length: 4, options: .storageModeShared)
    else { fail("could not allocate buffers") }

    inputData.withUnsafeBytes { raw in
        input.contents().copyMemory(from: raw.baseAddress!, byteCount: inputData.count)
    }
    memset(output.contents(), 0, elements * 4)
    scalar.contents().bindMemory(to: UInt32.self, capacity: 1)[0] = UInt32(elements)

    guard
        let queue = device.makeCommandQueue(),
        let buffer = queue.makeCommandBuffer(),
        let encoder = buffer.makeComputeCommandEncoder()
    else { fail("could not create a command encoder") }

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(output, offset: 0, index: 1)
    encoder.setBuffer(scalar, offset: 0, index: 2)
    let width = min(pipeline.threadExecutionWidth, threads)
    encoder.dispatchThreads(
        MTLSize(width: threads, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    encoder.endEncoding()
    buffer.commit()
    buffer.waitUntilCompleted()

    if let error = buffer.error { fail("dispatch failed: \(error)") }

    let produced = Data(bytes: output.contents(), count: elements * 4)
    try produced.write(to: URL(fileURLWithPath: arguments[5]))
    print("ran \(threads) threads from the archive")
} catch {
    fail("\(error)")
}
