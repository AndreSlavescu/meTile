window.BENCHMARK_DATA = {
  "lastUpdate": 1785318865514,
  "repoUrl": "https://github.com/AndreSlavescu/meTile",
  "entries": {
    "meTile Kernel Performance": [
      {
        "commit": {
          "author": {
            "email": "andre.slavescu@gmail.com",
            "name": "AndreSlavescu",
            "username": "AndreSlavescu"
          },
          "committer": {
            "email": "andre.slavescu@gmail.com",
            "name": "AndreSlavescu",
            "username": "AndreSlavescu"
          },
          "distinct": true,
          "id": "b0d63a5e5feb0e17c9d60350e6c9051cb7f29f57",
          "message": "write permissions",
          "timestamp": "2026-03-16T19:49:23-04:00",
          "tree_id": "87bf50d76f45917785d0d3c032cea132bb7cd5fd",
          "url": "https://github.com/AndreSlavescu/meTile/commit/b0d63a5e5feb0e17c9d60350e6c9051cb7f29f57"
        },
        "date": 1773705013918,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 541.71,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 3913.5,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 285.5,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 958.58,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 276.6,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 924.12,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 219.5,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 348.83,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 334.04,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 424.44,
            "unit": "us"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andre.slavescu@gmail.com",
            "name": "AndreSlavescu",
            "username": "AndreSlavescu"
          },
          "committer": {
            "email": "andre.slavescu@gmail.com",
            "name": "AndreSlavescu",
            "username": "AndreSlavescu"
          },
          "distinct": true,
          "id": "55ceb4dfd8ca3bd22db016f9cd3a35ddd898abf2",
          "message": "contributing.md + performance dashboard",
          "timestamp": "2026-03-16T19:59:09-04:00",
          "tree_id": "083cff33e140156bc3eb32684120250964cee976",
          "url": "https://github.com/AndreSlavescu/meTile/commit/55ceb4dfd8ca3bd22db016f9cd3a35ddd898abf2"
        },
        "date": 1773705595128,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 381.13,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 3458.13,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 366.96,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1110.79,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 466.15,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1084.98,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 275.83,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 239.88,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 251.98,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 351.33,
            "unit": "us"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andre.slavescu@gmail.com",
            "name": "AndreSlavescu",
            "username": "AndreSlavescu"
          },
          "committer": {
            "email": "andre.slavescu@gmail.com",
            "name": "AndreSlavescu",
            "username": "AndreSlavescu"
          },
          "distinct": true,
          "id": "944ebda5b59c5204b5be182a697b932330145809",
          "message": "improve regression timing",
          "timestamp": "2026-03-16T20:10:47-04:00",
          "tree_id": "3712321cda8001b73263c7c7a70f1a22bd0fae7a",
          "url": "https://github.com/AndreSlavescu/meTile/commit/944ebda5b59c5204b5be182a697b932330145809"
        },
        "date": 1773706310182,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 382.65,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 3308.54,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 328.06,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1098.21,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 329.08,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1130.83,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 289.81,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 303.08,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 296.21,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 394.21,
            "unit": "us"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "51034490+AndreSlavescu@users.noreply.github.com",
            "name": "Andre Slavescu",
            "username": "AndreSlavescu"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f69ab5917f4dad51c8a1c3abcba04aff849769c2",
          "message": "Merge pull request #5 from AndreSlavescu/ci\n\nadd pull-request write access for benchmark action",
          "timestamp": "2026-03-18T02:09:24-04:00",
          "tree_id": "daf27aa7b86b7f201915a50f2d40b589d5c831a9",
          "url": "https://github.com/AndreSlavescu/meTile/commit/f69ab5917f4dad51c8a1c3abcba04aff849769c2"
        },
        "date": 1773814233498,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 645.92,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 3938.52,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 286.46,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1021.02,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 290.48,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1179.21,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 264.98,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 272.31,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 283.25,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 411.19,
            "unit": "us"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andre.slavescu@gmail.com",
            "name": "AndreSlavescu",
            "username": "AndreSlavescu"
          },
          "committer": {
            "email": "andre.slavescu@gmail.com",
            "name": "AndreSlavescu",
            "username": "AndreSlavescu"
          },
          "distinct": true,
          "id": "4445e8b0bd4d6688e13008dd9fe03b010a5e2521",
          "message": "improve regression testing",
          "timestamp": "2026-03-18T02:14:11-04:00",
          "tree_id": "8c311359aaad50942ed701f6401b0a10b280064f",
          "url": "https://github.com/AndreSlavescu/meTile/commit/4445e8b0bd4d6688e13008dd9fe03b010a5e2521"
        },
        "date": 1773814545182,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 609.65,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 3763.4,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 496.56,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1243.57,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 519.14,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1089.58,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 397.93,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 340.33,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 346.79,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 512.38,
            "unit": "us"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "51034490+AndreSlavescu@users.noreply.github.com",
            "name": "Andre Slavescu",
            "username": "AndreSlavescu"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "65c69768073a40ccdb1b3c415f8bae75d8c8fc75",
          "message": "mlp fused kernel + compiler improvements (#4)\n\n* mlp fused kernel + compiler improvements\n\n* remove header\n\n* non-constant scalar coercion + max / min epilogue emission",
          "timestamp": "2026-03-18T02:47:07-04:00",
          "tree_id": "53b5230a6afcb7bb5e8091bcd71cc3b8b4817af3",
          "url": "https://github.com/AndreSlavescu/meTile/commit/65c69768073a40ccdb1b3c415f8bae75d8c8fc75"
        },
        "date": 1773816525282,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 411.45,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 3686.95,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 323.12,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1120.3,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 341.65,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1090,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 292.93,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 317.82,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 294.19,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 386.85,
            "unit": "us"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andre.slavescu@gmail.com",
            "name": "AndreSlavescu",
            "username": "AndreSlavescu"
          },
          "committer": {
            "email": "andre.slavescu@gmail.com",
            "name": "AndreSlavescu",
            "username": "AndreSlavescu"
          },
          "distinct": true,
          "id": "82d40642a9936602db6843d0ddd2539d2323408b",
          "message": "improve codegen",
          "timestamp": "2026-03-20T03:43:24-04:00",
          "tree_id": "3dda6ead276680fecf6fde382f8892ea689ff14d",
          "url": "https://github.com/AndreSlavescu/meTile/commit/82d40642a9936602db6843d0ddd2539d2323408b"
        },
        "date": 1773992701555,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 470.84,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 2993.71,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 363.27,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1209.23,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 396.11,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1187.04,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 274.62,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 304.2,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 317,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 443.99,
            "unit": "us"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andre.slavescu@gmail.com",
            "name": "AndreSlavescu",
            "username": "AndreSlavescu"
          },
          "committer": {
            "email": "andre.slavescu@gmail.com",
            "name": "AndreSlavescu",
            "username": "AndreSlavescu"
          },
          "distinct": true,
          "id": "451ddab05c2ccebd284263adfd1e14fa13581a06",
          "message": "format",
          "timestamp": "2026-03-20T03:45:25-04:00",
          "tree_id": "95108597a0e4719b424443f2dee7cf0b19bfbe71",
          "url": "https://github.com/AndreSlavescu/meTile/commit/451ddab05c2ccebd284263adfd1e14fa13581a06"
        },
        "date": 1773992819168,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 333.42,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 3383.83,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 368.55,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1097.26,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 296.14,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1061.08,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 248.57,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 261.38,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 266.07,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 350.17,
            "unit": "us"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "51034490+AndreSlavescu@users.noreply.github.com",
            "name": "Andre Slavescu",
            "username": "AndreSlavescu"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1626239f33a61c6f265944c6fadad8199527327d",
          "message": "Docs v1 (#6)\n\ndocs commit",
          "timestamp": "2026-03-20T04:02:00-04:00",
          "tree_id": "92553b3b4fff5d02ce9bbd5079050f33fb132d43",
          "url": "https://github.com/AndreSlavescu/meTile/commit/1626239f33a61c6f265944c6fadad8199527327d"
        },
        "date": 1773993819641,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 369.95,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 3442.44,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 339.67,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1103.04,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 351.56,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1131.62,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 257.99,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 307.14,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 336.87,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 416.22,
            "unit": "us"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "51034490+AndreSlavescu@users.noreply.github.com",
            "name": "Andre Slavescu",
            "username": "AndreSlavescu"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "719726bb5c8a5b634e8846e0de6315131a6e7939",
          "message": "Docs updates with diagrams (#7)\n\ndocs updates with diagrams",
          "timestamp": "2026-03-20T15:05:32-04:00",
          "tree_id": "d0fd48f1918d55dea0e2f75d570cd96d6817b56c",
          "url": "https://github.com/AndreSlavescu/meTile/commit/719726bb5c8a5b634e8846e0de6315131a6e7939"
        },
        "date": 1774033633997,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 384.28,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 2337.05,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 358.83,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1002.08,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 361.37,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1009.82,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 300.45,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 316.61,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 315.37,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 429.82,
            "unit": "us"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "51034490+AndreSlavescu@users.noreply.github.com",
            "name": "Andre Slavescu",
            "username": "AndreSlavescu"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "002a78f5e31d4be715ab1e000fa8a529505a1bcb",
          "message": "Add composable Metal compiler and guarded MLX runtime (#8)\n\nAdd composable compiler IR, measured schedule selection, proof-carrying graph rewrites, guarded MLX-LM integration, honest matched-precision benchmarks, and updated architecture documentation.",
          "timestamp": "2026-07-19T15:52:37-07:00",
          "tree_id": "0fe730186ff820ef294133d158d3e58650012496",
          "url": "https://github.com/AndreSlavescu/meTile/commit/002a78f5e31d4be715ab1e000fa8a529505a1bcb"
        },
        "date": 1784501731534,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 516.81,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 3860.28,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 378.29,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1118.33,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 334.36,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1171.44,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 348.81,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 355.38,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 420.07,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 422.8,
            "unit": "us"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "51034490+AndreSlavescu@users.noreply.github.com",
            "name": "Andre Slavescu",
            "username": "AndreSlavescu"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4824a75276fe64c38418b8cc67e67cce5dd1212a",
          "message": "Multi-row decode, algorithmic discovery, and honest matched-representation benchmarks (#9)\n\n* Add multi-row decode QMV and fix tournament measurement\n\nServe rows 2-31 in the dense BF16 decode path. MDotAccumulate and\nMPairedDotAccumulate now carry a `rows` dimension so each weight fragment is\nloaded once and reused across every activation row, keeping weight traffic flat\nwhile the work scales. MLX falls from ~110 GB/s at one row to 47-76 GB/s for\nrows 2-31; the multi-row QMV holds 105-122 GB/s, so the full MLP block runs\n1.5-1.8x faster there and the down projection alone up to 2.4x. This is the band\na speculative-decode verification pass runs in.\n\nFor rows > 1 the tuner requires every row to equal MLX's own single-row result\nrather than its multi-row tile kernel, so a batched step stays bit-identical to\ndecoding those tokens one at a time.\n\nFix two defects that let the tournaments select kernels slower than native MLX:\n\n- Every tuner timed one dispatch per sample. The blocking mx.eval round trip\n  costs ~200 us whatever the kernel does, so it was added to each candidate and\n  compressed their ratios toward 1.0, letting switch margins admit losers.\n  calibrate_tournament_batch now sizes a batch per timed sample, and the\n  measurement source joins each backend signature so stale picks invalidate.\n- _COMPILED_SWITCH_MARGIN was 0.005, below the run-to-run noise floor, so the\n  mx.compile variant won on noise and then measured 0.63-0.90x in steady state.\n\nMeasured at matched representation, BF16 rows=128 went 0.82x -> 1.11x and every\nINT4 cell recovered to parity.\n\nSkip candidate filtering on schedule-cache hits: it rebuilt the filtered config\ntuple on every dispatch, 70% of the Python cost of a decode call.\n\nAdd benchmarks/matched_representation_matrix.py, which runs identical weights in\nidentical formats on both sides so a result is a kernel comparison rather than a\nrepresentation change. Replace the model charts with three that report one unit,\na multiplier against native MLX, and graph only matched-representation runs. The\nBF16 capacity suite compares meTile INT8 decode against MLX BF16, so it is no\nlonger plotted; its numbers stay in the README table and committed JSON.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n(cherry picked from commit c782369f17aa350b6fbdcad693d07ca51eeb16c2)\n\n* Consolidate algorithmic discovery and chart what actually wins\n\nFold attention_discovery into algo_discovery so one module covers every rewrite the\ncompiler can find. Both rewrites are licensed by the same proven law: the\nweighted-softmax monoid is proved once, flash attention uses all three components and\nonline softmax the (maximum, normalizer) projection with the value fixed at 1. Adding\na rewrite is now a matcher, not a new proof. Reduction laws and certificate types are\nre-exported, so composing discovery needs one import.\n\nRoute discovery through the exact min-cut selector graph_fusion already used. Overlapping\nrewrites are mutually exclusive, so choosing between them is maximum-weight independent\nset, and the project-selection reduction solves it exactly on bipartite conflict\ncomponents. find_flash_attention previously took whichever region it hit first.\n\nMake the online softmax rewrite work; it is now on by default and 8/8. Three fixes:\n\n- It reused the tile-max chain's Constant as the running scalar maximum, but lowering\n  promotes that seed to a tile-typed accumulator. Two dedicated scalar seeds instead,\n  and both post-loop reductions dropped, since re-reducing an already threadgroup-wide\n  value inflates the normalizer by the thread count.\n- _emit_vec4_op fell through for MThreadgroupReduce and emitted simd_max(float4).\n- A threadgroup reduction cannot sit inside the ragged tail's mask branch, because\n  masked threads never reach the barrier. ForRange.masked_identity plus\n  _predicate_masked_load guard only the load and seed the value with an identity the\n  law supplies: -inf is the identity of the maximum, and exp(-inf - m) = 0 is the\n  identity of the normalizer, so one value serves both reductions.\n\nMeasured 1.28x against a 1.33x four-transfers-to-three ceiling. It is a compiler result,\nnot a competitive one: MLX's softmax is a single fused kernel and still 0.74x to 0.99x\nfaster than ours.\n\nFix a crash the multi-row tuner introduced. MLX-LM passes rank-3 [batch, sequence,\nhidden], and slicing axis 0 for the per-row reference yields empty rows once batch is\nsmaller than the row count, which segfaults inside eval. Rows now come from a flattened\nview, and the reference is only built in the QMV band where it is actually checked.\n\nStop materializing a causal bias tensor in the attention tournament's native reference.\nMLX masks causally without allocating anything and the two agree bitwise, including for\nragged query counts. Building the bias made the baseline look slower than it is and\nbiased selection toward the generated kernel; causal went 0.74x to 1.01x.\n\nAdd three affine tilings for wide outputs. Sweeping the full legal space found the\nshipped six topped out at parity for N >= 8192 while 64x128 hilbert reaches 1.10x.\n\nInclude the measurement source in the framework and attention persistent keys, so\nchanging how candidates are timed invalidates stored picks instead of leaving stale ones.\n\nAdd benchmarks for shape sensitivity, graph fusion, and mx.compile comparison, and chart\nthe two findings that explain the model results: MLX changes kernel between output widths\n2048 and 2560 and the one below is poor, which is why Llama 3.2 1B gains 3.18x and 3B only\n1.08x; and batching should hold weight bandwidth flat, which meTile does for BF16 and\nnothing yet does for INT4 or INT8.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n(cherry picked from commit b2b5fa27ec0185db926e3b1c26c9168ec29f5e94)\n\n* Refresh benchmark data and shorten chart subtitles\n\nRe-measure the batch-size matrix and shape sensitivity at HEAD so the charts, the\nJSON, and the README all quote the same run. Two results moved:\n\n- BF16 rows=512 went 0.85x to 0.99x. The earlier dip was the tournament picking a\n  losing schedule, not a kernel limit, so the chart no longer shows a regression\n  that does not exist.\n- INT4 at width 8192 went 1.01x to 1.10x, now that the wide-output tilings added in\n  the previous commit are in the candidate list.\n\nCut every chart subtitle to one short line. They were carrying explanation that\nbelongs in prose, and the batch chart's ran off the canvas.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n(cherry picked from commit 759969c71a8c83816f2bfd8bcab62df316d0ef73)\n\n* Measure meTile's quantized path instead of assuming it defers\n\nThe batch chart drew MLX INT4 and INT8 with no meTile counterpart, which reads as\nmissing data rather than as a result. The benchmark was recording None for those\npoints on the grounds that meTile has no multi-row quantized kernel, which was an\nassumption rather than a measurement.\n\nRun the quantized executor path and record it. It does track MLX, within 1.0% for\nINT4 and 5.5% for INT8, so the label now says meTile defers to MLX rather than\nleaving a gap.\n\nTime the batch sweep interleaved as well. Measuring one side after the other put\nINT8 at 76 GB/s against MLX's 116 at one row, which contradicted the batch-size\nmatrix; alternating the two removes it and the formats agree.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n(cherry picked from commit 67d0ca936ceceb552c747ab73d09b2bccbeb4259)\n\n* Label the shared quantized lines plainly\n\nFor INT4 and INT8 meTile calls MLX's kernel rather than running one of its own, so a\nsingle line describes both backends. \"meTile defers to MLX\" explained an implementation\ndetail; \"INT4, both\" says what the line is.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n(cherry picked from commit ea1ea1e005aa0df1011a09d32e08dc78ccfaaf98)\n\n* Fix lint: import calibrate_tournament_batch, reformat at the project width\n\nThe lint job caught three things the test job could not.\n\nmlx_block_scaled.py called calibrate_tournament_batch without importing it, so\nblock-scaled tuning raised NameError as soon as it reached the tournament. Every\nother backend imports it; this one was missed. No test covers that path.\n\nThe rest is mechanical: I had hand-wrapped near 88 columns while the project\nformats at 100, and the redundant \"# noqa: E402\" markers in benchmarks are\nalready covered by per-file-ignores.\n\nB023 is ignored for benchmarks rather than silenced per line. The timing helpers\ntake the work as a closure and call it before returning, so a closure built in a\nsweep loop never outlives the iteration that built it, and none are stored in a\ncontainer. Binding each captured tensor as a default argument would only obscure\nwhat is being measured.\n\n582 tests pass; ruff format, ruff check, and vulture are clean.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* Give the int4 decode MLP a candidate that works above one row\n\nBoth fused affine paths are single-row by construction: the SwiGLU kernels\nraise \"native affine SwiGLU schedules require one decode row\" and the residual\nQMV raises its equivalent. So from two rows up the tournament had only the\nscalar kernels, which lose, and it settled on native MLX for the entire batched\nband. That is why the matrix reports int4 at parity there.\n\nmeTile already generates a multi-row affine matmul, and on a single projection\nit holds ~60 GB/s flat across rows 8 to 16 while MLX falls from 68 to 35. This\nadds two candidates built from it: a SwiGLU composed of two multi-row matmuls,\nand a down projection that adds the residual afterwards.\n\nNothing is forced. Both go through the same tournament and the same numerical\ncompatibility gate as every other candidate, and from_mlx rejects anything but\n4-bit group-64 so they cannot compete at formats they do not support.\n\nMeasured on the MLP block 1536 -> 8960 -> 1536, interleaved against native MLX:\n\n  rows      1     2     4     8    16    32\n  before  0.92  0.98  0.94  1.03  1.00  1.04\n  after   0.95  0.96  1.01  1.28  1.31  1.24\n\nRows 1 to 4 are unchanged within noise and still pick native. Those sit below\n1.0 in both columns, which is the pre-existing mlx_compiled selection problem,\nnot something this introduces.\n\n585 tests pass.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* Confirm dense SwiGLU finalists head to head, not in a crowded rotation\n\nHow long a candidate measures depends on how many others share the round-robin,\nso ranking finalists from one big rotation can prefer a kernel that loses when\nthe two are timed against each other. The holdout that re-measured in a small\ngroup only ran for one-row decode; every multi-row shape committed to whatever\nthe crowded rotation ranked first.\n\nEach finalist is now timed in its own two-way round-robin against native, and\nranked on the ratio to native rather than the raw time, so drift between one\npairing and the next cancels.\n\nDense SwiGLU 1536 -> 8960, paired against native MLX:\n\n  rows      2     4     8    16\n  before  1.43  1.40  1.40  1.42\n  after   1.49  1.48  1.48  1.47\n\nSelections also settle down, picking the same shape across neighbouring row\ncounts instead of a different one each time.\n\nTwo things tried and rejected, recorded so they are not retried blind:\n\nRaising _MAX_QMV_ACCUMULATOR_PAIRS from 16 to 32 does nothing. The bound looked\ntoo tight because outputs_per_simdgroup=2 at 16 rows compiles to 115 registers\nagainst G17's 140-register budget, so it cannot be spilling. But the tuner\nalready reaches the same speed with outputs_per_simdgroup=1 at a larger\nsimdgroup count, and the wider search bought no speed. Left at 16.\n\nThe 1.28x I expected to recover here was not real. It came from comparing the\ntuner's pick against the best row of a partial hand-built table that omitted the\nsimdgroup counts the tuner actually explores. Measured against the tuner's real\nbaseline the gain is the 3 to 6% above.\n\n586 tests pass.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* Read register counts out of compiled Metal kernels\n\nMetal exposes no register count, and the obvious stand-in does not work:\nmaxTotalThreadsPerThreadgroup reads 1024 on an M5 whether a kernel holds 4 live\nfloats or 512. The compiler does record the number, in a metadata segment of the\nGPU binary, and MTLBinaryArchive.serialize is the only public way to get that\nbinary onto disk.\n\nThree unwraps to reach it. The archive is a fat file whose applegpu_* slice is\nthe GPU code; that slice's __compute section is itself a Mach-O; inside it\n__GPU_METADATA is a FlatBuffer and __text is the machine code. The count is\nfield 0 of the table referenced by field 0 of the root.\n\nRead by path, never by byte offset. The buffer embeds the kernel name and\nsignature, so a fixed offset drifts between kernels: reading byte 188 worked for\none probe kernel and silently reported a neighbouring field, 24 registers and\n\"spilling\", for every real one. --self-check pins the reader against known\ncounts so a bad read fails loudly.\n\nWhat it establishes on G17: the budget is 140 registers per thread, the count\ntracks live floats plus 4 up to 124, and kernels that reach 140 are spilling and\nmeasured 1.3x to 6.7x slower than lower-register siblings.\n\nApplied to the dense SwiGLU bound it says the bound is already right: every\nconfig admitted by _candidate_configs peaks at 99 of 140 registers, 71% of\nbudget, so nothing admitted can spill. That is the intended use, checking a\nscheduling bound rather than trusting it, and it is why the tool ships despite\nhaving produced no speedup of its own.\n\nAnalysis tool only. Each reading costs a Metal compile of about a second, far\ntoo slow for the tuner to call, and it degrades to a message rather than an\nerror when swiftc is absent.\n\n586 tests pass.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* Lift the tuning machinery out of the MLX backend\n\nThe compiler core has never depended on MLX: frontend, ir, compiler, codegen and\nruntime are 17k lines with no mlx import, and metile/runtime/metal_device.py\nreaches Metal directly through ctypes, which is how the gemm, softmax, rmsnorm,\nattention, fft and mlp benchmarks execute. MLX is a comparison target there.\n\nWhat had drifted is that the machinery for choosing between kernels grew inside\nmetile/backends/mlx*.py, where eight files each carry their own round-robin,\nswitch margins and cache handling. None of that is MLX-specific. A second\nexecution backend would have to duplicate all of it.\n\nmetile/tuning now holds the backend-agnostic part: round_robin for cheap triage\nof a large field, confirm_pairwise for deciding, select_fastest for the margin\nand tie-break policy, token_bucket for shape bucketing. It knows nothing about\nkernels or how to run one; callers pass a measure function that turns a thunk\ninto seconds. mlx_dense_swiglu is ported onto it as the first consumer and its\nbehaviour is unchanged. The remaining seven backends still hold their own copies.\n\nCorrecting the previous commit while it is still unmerged: it claimed pairwise\nconfirmation was worth 3 to 6%, reading 1.40-1.43x going to 1.45-1.49x. That was\na trend read out of variance. Interleaved against the pre-change commit, four\nalternating pairs, medians by row count:\n\n  rows      2     4     8    16\n  before  1.42  1.48  1.45  1.49\n  after   1.48  1.45  1.45  1.53\n\nMixed, and inside a run-to-run spread of about 0.07. There is no measurable\nspeedup. The reason to keep pairwise confirmation is that it makes selection\nprincipled rather than fast: every candidate is measured in an identically sized\ncontext, so the ranking no longer depends on how crowded the rotation happens to\nbe. Ranking on the ratio to the baseline is what makes separate pairings\ncomparable when the baseline drifts.\n\n587 tests pass.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-29T04:01:02-04:00",
          "tree_id": "0723fcf03106619ff1f062c67dc9f3f450675246",
          "url": "https://github.com/AndreSlavescu/meTile/commit/4824a75276fe64c38418b8cc67e67cce5dd1212a"
        },
        "date": 1785312269118,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 460.63,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 4328.08,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 455.49,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1412.13,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 474.29,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1413.46,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 356.39,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 367.21,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 384.26,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 446.96,
            "unit": "us"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "51034490+AndreSlavescu@users.noreply.github.com",
            "name": "Andre Slavescu",
            "username": "AndreSlavescu"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "76d18327b829c28599c107f2ed05655dabe1301e",
          "message": "Measure each model at its own layer shapes (#10)\n\nQwen3.5-4B and Qwen3.5-9B both benchmark at exactly 1.000x end to end, and that\nnumber on its own is misleading. It says nothing gained, when what is true is\nthat nothing was available at the one shape the model-level harness exercises.\n\nA transformer is one shape repeated, so measuring that shape separates the two\neffects an end-to-end figure conflates. Width decides prefill: MLX's int4 kernel\nis weak below an output width of about 2560, and only the down projection can\nland there, because gate and up always output `intermediate`, which is wide in\nevery model measured. Batch decides decode: MLX re-reads weights per row tile\nabove one row, whatever the width.\n\nMeasured on M5, int4 group 64, identical weights on both sides:\n\n  model             hidden  inter   pre up  pre down  rows 1  rows 8  rows 16\n  Qwen2.5 0.5B         896   4864    0.99x     2.27x   0.98x   1.23x    1.52x\n  Qwen2.5 1.5B        1536   8960    1.07x     2.84x   1.01x   1.26x    1.31x\n  Llama 3.2 1B        2048   8192    1.05x     3.11x   0.98x   1.26x    1.79x\n  Llama 3.2 3B        3072   8192    1.07x     1.07x   0.97x   1.23x    1.78x\n  Qwen3.5 4B          2560   9216    1.11x     1.23x   1.02x   1.21x    1.27x\n  Qwen3.5 9B          4096  12288    1.07x     1.15x   0.98x   1.21x    1.24x\n\nSo the newer models gain nothing at prefill because they have no layer below the\ncliff, and nothing at single-token decode because that is bandwidth bound. They\ndo gain 1.21x to 1.27x once more than one row is in flight, which the end-to-end\nfigure never reaches.\n\nShapes are read from the local Hugging Face cache rather than hardcoded, and\nmultimodal checkpoints have their language model under text_config, which both\nQwen3.5 models are.\n\nAn earlier version of this measured prefill on the gate projection only and\nreported 1.02x to 1.10x across the board, hiding the cliff entirely: gate\noutputs `intermediate`, which is above the cliff in every model here. Both\ndirections are now reported so the comparison cannot be read the wrong way.\n\n587 tests pass.\n\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-29T04:04:22-04:00",
          "tree_id": "1428f9284315d142a23fe0957623e81279d20beb",
          "url": "https://github.com/AndreSlavescu/meTile/commit/76d18327b829c28599c107f2ed05655dabe1301e"
        },
        "date": 1785312452050,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 542.06,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 4255.71,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 473.61,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1346.36,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 492.18,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1393.51,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 365.27,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 407.06,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 456.55,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 463.05,
            "unit": "us"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "51034490+AndreSlavescu@users.noreply.github.com",
            "name": "Andre Slavescu",
            "username": "AndreSlavescu"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0d3d885c93c0782311660061cafb08e8a92ca7e2",
          "message": "Measure how much instruction scheduling can buy on this GPU (#11)\n\nReordering instructions only pays if the hardware stalls without independent\nwork nearby, and that is a property of the machine, not of a kernel. This asks\nit directly: dependent fma chains, no memory traffic, replicated into N\nindependent chains. The ratio between one chain and saturation is the most any\nscheduler could ever win here.\n\nOn M5 (G17):\n\n  chains   fp32 GFLOP/s   vs 1     fp16 GFLOP/s   vs 1\n       1           3740   1.00x            4596   1.00x\n       2           4072   1.09x            5964   1.30x\n       4           4086   1.09x            6337   1.38x\n      12           4082   1.09x            6485   1.41x\n\nA single dependent chain already reaches 92% of fp32 peak. The GPU covers\nlatency with thread-level parallelism rather than instruction-level parallelism\ninside a thread, so fp32 saturates at two chains and never improves again.\n\nThe whole payoff available to instruction scheduling on this hardware is\ntherefore 1.09x on fp32 and 1.41x on fp16, and only for compute-bound code. That\nexplains why five scheduling experiments on the int4 QMV all came back flat:\nunroll factors of 1, 2, 3 and 6 measured 50.2 to 50.9 GB/s, indistinguishable.\nMemory-bound kernels get none of it.\n\nThe same run gives the number that does matter. Scalar peak is 4.1 TFLOP/s fp32\nand 6.5 fp16, against a matrix-unit peak of 15.3. Choosing the right functional\nunit is worth 2.4x where scheduling is worth 1.09x, so compiler effort belongs in\nmatrix-unit tiling.\n\nInterleaved rather than swept, because these kernels run for milliseconds and a\nsequential sweep measures thermal drift as much as it measures the kernels.\n\n587 tests pass.\n\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-29T04:26:20-04:00",
          "tree_id": "c9ecc088599fd54ce3d73fc024fc9b0a798679f6",
          "url": "https://github.com/AndreSlavescu/meTile/commit/0d3d885c93c0782311660061cafb08e8a92ca7e2"
        },
        "date": 1785313785102,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 586.12,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 4337.08,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 414.86,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1342.64,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 380.34,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1321.32,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 344.21,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 383.58,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 418.29,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 549.61,
            "unit": "us"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "51034490+AndreSlavescu@users.noreply.github.com",
            "name": "Andre Slavescu",
            "username": "AndreSlavescu"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "886b758408f339c3f61681d6a028ef4c4a1d8ac4",
          "message": "INT4 is no longer parity above four rows, and the published numbers now say so (#12)\n\n* Stop selecting mx.compile for affine SwiGLU, and share the tuning machinery\n\nTwo problems, one cause. The int4 decode block measured 0.919x against native\nMLX at one row, and the tuner was choosing the mx.compile variant to get there.\n\nMeasured directly, interleaved and batched, mx.compile of the affine SwiGLU is\n0.938x eager at one row, 0.946x at two, 1.014x at four and 1.005x at eight. It\nis never faster than noise and clearly slower exactly where it kept being\nselected. _COMPILED_SWITCH_MARGIN had already been raised twice to stop this and\ndid not, so the candidate is withdrawn rather than margined against again.\n\nThe reason it kept winning is the same one fixed for dense SwiGLU: a candidate's\nmeasured time depends on how many others share the round-robin, so ranking from\none crowded rotation does not survive head-to-head measurement. The affine tuner\nnow confirms finalists pairwise against native through metile/tuning.\n\nint4 decode block, 1536 -> 8960 -> 1536, paired against native MLX:\n\n  rows       1     2     4     8    16    32\n  before  0.93  0.97  1.00  1.22  1.30  1.25\n  after   1.04  1.01  1.00  1.24  1.32  1.24\n\nNothing is now slower than native at any row count.\n\nAlso ports mlx_dense and mlx_dense_residual onto metile/tuning. Both carried\nbyte-identical copies of the round-robin and the switch-margin logic, and the\nshared batched_measure now lives beside calibrate_tournament_batch where the\nreasoning about the eval round trip already was. Four of eight backends are on\nthe shared layer; mlx.py, mlx_affine, mlx_attention and mlx_block_scaled still\nhold their own copies.\n\n587 tests pass.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* Put the affine and block-scaled tuners on the shared confirmation path\n\nBoth carried their own copy of the round-robin, and both decided from a single\ncrowded rotation. That is the ranking that does not survive isolated\nmeasurement: how long a candidate reads depends on how many others share the\nrotation with it.\n\nAffine now confirms finalists pairwise against native, the same shape the dense\nand quantized tuners use. Block-scaled has no native candidate to pair against,\nso the provisional fastest serves as the reference; what matters there is not\nwhich kernel is the baseline but that every finalist is measured in an\nidentically sized context.\n\nSix of eight MLX backends now share metile/tuning. mlx.py and mlx_attention\nstill hold their own copies.\n\nBoth tuners key their persistent caches on the tuner source, so existing\nselections invalidate on their own rather than surviving the change.\n\n587 tests pass.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* Finish moving every MLX backend onto the shared tuning layer\n\nmlx.py's framework tuner and mlx_attention were the last two carrying their own\nround-robin, and mlx_dense_swiglu still had a private copy of the batched timing\nhelper. All eight backends now measure through metile/tuning and time through a\nsingle batched_measure that sits beside calibrate_tournament_batch, where the\nreasoning about the eval round trip already lived.\n\nBoth tuners also gain pairwise confirmation, so finalists are decided head to\nhead rather than from a rotation whose size changes what each candidate measures.\n\nSelection policy is deliberately left alone in these two. _choose_framework_config\nruns choose_mdl_tie over every generated candidate rather than over a cutoff\ncluster, which is not what select_fastest does, so porting the measurement without\nthe selection keeps behaviour identical where it was not the thing being fixed.\n\nmlx_attention carries four-element candidates rather than three; round_robin and\nconfirm_pairwise only read the identity and the thunk, so they take either.\n\n587 tests pass, vulture clean.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* Refresh the published matrix: int4 is no longer parity above four rows\n\nThe matched-representation table predated the multi-row int4 candidates and the\nselection fix, so it reported the project as slower than it is. Re-measured on\nthe same machine, 25 interleaved rounds:\n\n  rows          1     2     4     8    16    32   128\n  BF16       1.02  1.69  1.82  1.65  1.52  1.06  1.11\n  INT4       1.02  1.02  1.02  1.29  1.31  1.23  1.00\n  INT8       0.98  0.99  1.00  1.00  1.00  1.00  1.00\n\nint4 was published as 0.99 / 0.88 / 0.99 / 1.00 / 0.99 / 0.98 across those row\ncounts. The 0.88 at two rows is gone with the mx.compile candidate, and rows 8\nto 32 now win outright at 100% of rounds rather than tying.\n\nThe batch-efficiency chart carried the same staleness in a worse form: int4 drew\none line labelled \"both\" on the assumption meTile defers to MLX there, which is\nno longer true, and the renderer had no path to draw two lines for a format that\nstops tracking. It now encodes format as colour and backend as dash, so each\npair sits in one hue and int4's gap is visible (70.9 against 52.4 GB/s at eight\nrows, 44.5 against 33.8 at sixteen). Palette checked with the validator: all\nchecks pass, and the one contrast warning is relieved by the direct labels that\nwere already there. The legend no longer repeats those labels six times and\nstates the encoding instead.\n\nTwo framing corrections while the numbers were being redone. \"Weights are read\nonce, so these should stay flat\" is wrong past eight rows, where both backends\nslope down because the same weights are serving eight to thirty-two times the\narithmetic; that part is not waste and the chart no longer implies it is. And\nsingle-token decode was described as MLX running at 120 to 126 GB/s against a\n120 GB/s ceiling, which cannot be right. Measured with batched dispatches it is\n93 to 97% of the ceiling, and a hand-written kernel matches without beating it.\n\n587 tests pass.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-29T05:51:38-04:00",
          "tree_id": "597c75c13d216090b322ce0b95adfa6725d7c257",
          "url": "https://github.com/AndreSlavescu/meTile/commit/886b758408f339c3f61681d6a028ef4c4a1d8ac4"
        },
        "date": 1785318863742,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 445.22,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 3837.34,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 341.6,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1182.24,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 343.47,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1214.8,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 260.39,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 272.27,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 288.9,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 349.02,
            "unit": "us"
          }
        ]
      }
    ]
  }
}