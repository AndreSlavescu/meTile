// Compile MSL to a native GPU binary archive so its machine code and metadata can be read.
//
// usage: agxdump <source.metal> <functionName> <out.bin>
//
// Metal exposes no way to see a compiled kernel's register usage. The compiler does record
// it, in a metadata segment of the GPU binary, and MTLBinaryArchive.serialize is the only
// public way to get that binary onto disk. Everything downstream is unwrapping.

import Foundation
import Metal

let arguments = CommandLine.arguments
guard arguments.count == 4 else {
    FileHandle.standardError.write(
        "usage: agxdump <source.metal> <function> <out.bin>\n".data(using: .utf8)!)
    exit(2)
}

guard let device = MTLCreateSystemDefaultDevice() else {
    FileHandle.standardError.write("no Metal device\n".data(using: .utf8)!)
    exit(1)
}

do {
    let source = try String(contentsOfFile: arguments[1], encoding: .utf8)
    let library = try device.makeLibrary(source: source, options: MTLCompileOptions())
    guard let function = library.makeFunction(name: arguments[2]) else {
        FileHandle.standardError.write(
            "no function \(arguments[2]); available: \(library.functionNames)\n"
                .data(using: .utf8)!)
        exit(1)
    }

    let pipeline = try device.makeComputePipelineState(function: function)
    print("device: \(device.name)")
    print("maxTotalThreadsPerThreadgroup: \(pipeline.maxTotalThreadsPerThreadgroup)")
    print("threadExecutionWidth: \(pipeline.threadExecutionWidth)")
    print("staticThreadgroupMemoryLength: \(pipeline.staticThreadgroupMemoryLength)")

    let descriptor = MTLComputePipelineDescriptor()
    descriptor.computeFunction = function
    let archive = try device.makeBinaryArchive(descriptor: MTLBinaryArchiveDescriptor())
    try archive.addComputePipelineFunctions(descriptor: descriptor)
    try archive.serialize(to: URL(fileURLWithPath: arguments[3]))
} catch {
    FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
    exit(1)
}
