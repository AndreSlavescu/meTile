window.BENCHMARK_DATA = {
  "lastUpdate": 1785380240195,
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
          "id": "5dc233c11932be042380387a730aecdcf1ea9657",
          "message": "Chart the newer models, whose 1.000x was hiding a real result (#13)\n\nQwen3.5-4B and Qwen3.5-9B were benchmarked but never plotted, and no chart\ncovered the per-shape data at all. Both models report exactly 1.000x end to end,\nwhich reads as \"meTile does nothing for these\" when the truth is narrower:\nnothing was available at the one shape the model harness exercises.\n\nThe new chart plots the three measurements that separate the cases. At one row\nevery model sits at parity, because that is bandwidth bound. At sixteen rows\nevery model gains, including both Qwen3.5 checkpoints, because the weights get\nreused. Only prefill varies by model, and it varies with exactly one property:\n\n  model            hidden   prefill down   1 row   16 rows\n  Qwen2.5 0.5B        896          2.27x   0.98x     1.52x\n  Qwen2.5 1.5B       1536          2.84x   1.01x     1.31x\n  Llama 3.2 1B       2048          3.11x   0.98x     1.79x\n  Qwen3.5 4B         2560          1.23x   1.02x     1.27x\n  Llama 3.2 3B       3072          1.07x   0.97x     1.78x\n  Qwen3.5 9B         4096          1.15x   0.98x     1.24x\n\nThe three below 2560 win prefill outright; the three at or above it do not. That\nis the whole spread, and depth has nothing to do with it.\n\nPalette checked with the validator rather than eyeballed: all checks pass, and\nthe one contrast warning on the green is relieved by the per-point value labels.\nFirst render put sub-parity labels to the left of their points, where they landed\non top of the model names, so every label now sits to the right.\n\n587 tests pass.\n\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-29T14:33:30-04:00",
          "tree_id": "ad5883580e9f3cf7daefd970a5af131c8e6353ba",
          "url": "https://github.com/AndreSlavescu/meTile/commit/5dc233c11932be042380387a730aecdcf1ea9657"
        },
        "date": 1785350239237,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 914.99,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 4329.54,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 574.55,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1508.13,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 587.42,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1469.03,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 432.43,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 477.17,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 525.21,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 570.95,
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
          "id": "36272c1c06409603424c45ff59e86df899465163",
          "message": "Move the measured hardware model into the compiler (#14)\n\n* Do not switch away from native unless two measurements agree\n\nQwen3.6-27B's down projection was running at 0.79x native in the shape matrix.\nThe cause was not the kernel and not the tuner picking badly once. At the widest\nprefill shapes the measurement itself stops being able to rank candidates.\n\nMeasured at K=17408, the same three kernels, three ways:\n\n                    native   bn=128   bn=256\n  round-robin        2109     2701     1865    bn=256 fastest\n  pairwise           2090     3279     2433    bn=256 loses at 0.86x\n  isolated           2122     2486     1791    bn=256 wins at 1.18x\n\nThree readings, three orderings. Selection cannot be trusted there, and because\nthe answer is written to the persistent cache, one bad draw is replayed on every\nlater run. That is how a config measuring 0.85x native ended up serving the 27B.\nWith the cache disabled the same shape measures 1.00x; with it, 0.79x.\n\nThe tuner now requires the round-robin and the pairwise pass to agree, both\nclearing the switch margin, before it will leave native. Disagreement means the\nmeasurement cannot resolve the difference, and the honest response is to keep the\nkernel known not to lose. The gate only ever rejects candidates, so it cannot\nadmit anything new.\n\nThis is a mitigation, not a fix, and it should not be recorded as one. Across\nrepeated tunings it moved the 27B down projection from a cached 0.85x to native\nin four of five draws, but selections measuring 0.94x and 0.87x still get through\nat other shapes, and a small shape (K=1536) produced a 0.95x as well. The\nunderlying problem is that the verification measurement is itself noisy at these\nsizes, so no amount of agreement between two noisy passes guarantees a win.\n\nAlso releases each model's arrays in the shape matrix before quantizing the next.\nThat was investigated as the cause of the above and was not it, but holding two\nmodels' weights live across the transition is about a gigabyte of overlap at 27B\nscale and is worth not doing regardless.\n\n587 tests pass, vulture clean.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* Move the measured hardware model into the compiler\n\nThe AGX work lived in benchmarks/, which is the wrong home for it. A register\nbudget and an ILP ceiling are properties of the target, and a pass that wants to\nknow whether a schedule can spill, or whether reordering could possibly pay,\nshould be able to ask the compiler rather than read a comment quoting a number\nsomebody once measured.\n\nmetile/target/agx.py now holds the machine model and the binary inspection that\nproduces it, each value recorded with how it was obtained and what it settled:\n\n  REGISTER_BUDGET        140, found by growing live values until the count stopped\n                         rising, then confirming kernels reaching it spill\n  ILP_CEILING            1.09x fp32, 1.41x fp16, from dependent fma chains against\n                         independent ones. A single dependent chain already hits\n                         92% of fp32 peak, which is why scheduling is not where\n                         effort belongs on this target\n  SCALAR/MATRIX peak     4.1 and 6.5 against 15.33 TFLOP/s, the gap that makes\n                         functional-unit selection outrank scheduling\n  STREAMING_READ_GBPS    120.6, measured, not the 153 on the spec sheet\n\nbenchmarks/agx_registers.py and agx_ilp_ceiling.py become consumers: the first is\nnow purely the command line over the reader plus the audit of what the dense\nSwiGLU bound admits, which is the reason the tool exists.\n\nOne thing deliberately not done. The dense SwiGLU bound's comment cites the\nbudget, and importing REGISTER_BUDGET there to make the link look structural\nwould be dishonest: the bound is rows * outputs_per_simdgroup <= 16, and the\nregister count is measurably not a function of that product, so the constant\nwould be decorative. Lint caught the unused import and it is gone. The comment\nnames the module instead.\n\ntests/test_target.py guards what depends on the model's shape, including that an\nunknown element type reports no ILP headroom rather than inheriting the largest\nknown value, and that the matrix-to-scalar ratio still exceeds the ILP ceiling.\nIf that ordering inverts on new hardware, the guidance built on it needs\nrevisiting rather than silently carrying over.\n\n591 tests pass, vulture clean.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-29T18:55:56-04:00",
          "tree_id": "42ec3dca31233efb87043405c1b7ac029413d7f6",
          "url": "https://github.com/AndreSlavescu/meTile/commit/36272c1c06409603424c45ff59e86df899465163"
        },
        "date": 1785365946081,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 498.5,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 4047.81,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 435.42,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1263.64,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 403.11,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1171.26,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 295.63,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 296.85,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 385.28,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 407.42,
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
          "id": "d84518b33723c04db5115b6e278ba09bfd3b325e",
          "message": "Rank kernels by how they behave ordinarily, not at their best (#15)\n\nThis is the actual fix for the 0.79x the previous commit could only mitigate, and\nit came from measuring the thing rather than the ratio. At a 17408-wide affine\nmatmul the same three kernels, sampled 25 times in five independent rounds:\n\n              min       median\n  native     2038       2070\n  bn=128     1643       2874\n  bn=256     1633       2745\n\nThe generated kernels are bimodal. They are genuinely quicker than native at\ntheir best, by about 1.24x, and slower than it most of the time. Native is tight:\nits minimum and median are 2% apart.\n\nThat explains every confusing observation. The median-ranked tuner sometimes drew\na favourable sample and switched away from native. An isolated verification then\ndrew its own favourable sample and appeared to confirm 1.18x. Both were sampling\nthe fast mode of a kernel that does not usually run in it. The earlier conclusion\nthat \"the measurement cannot rank candidates at these shapes\" was wrong: the\nmeasurement is fine, the estimator was asking the wrong question.\n\nmetile.tuning.pessimistic summarises samples at the 75th percentile instead of\nthe median, so a candidate is judged on how it behaves when conditions are\nordinary rather than ideal. Where candidates are equally consistent it agrees\nwith the median, so it only changes decisions in the case it exists to catch.\n\nTwelve tunings across four shapes, three of which previously produced losing\nselections:\n\n  27B down    was 0.79x cached      now 1.18x, native, native\n  27B up      was 0.94x, 0.87x      now 1.09x, 1.01x, 1.10x\n  1.5B up     was 0.95x             now 1.04x, 1.06x, 1.08x\n  Llama-1B    genuine 3x win        now 3.20x, 3.21x, 3.25x\n\nNo losses, and the win that matters is untouched, which was the risk: a guard\nthis blunt could have thrown away the 3.2x along with the noise.\n\n593 tests pass.\n\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-29T19:00:43-04:00",
          "tree_id": "5dcf8313281c33177d840365039bc3f87b25dcc4",
          "url": "https://github.com/AndreSlavescu/meTile/commit/d84518b33723c04db5115b6e278ba09bfd3b325e"
        },
        "date": 1785366237779,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 442.76,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 3567.33,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 350.52,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1170.45,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 375.2,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1096.78,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 308.19,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 296.17,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 298.43,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 420.61,
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
          "id": "c5a9beea831aacfd9dd62fb21e0391bb7fb6a94a",
          "message": "Publish nine models, Qwen3.6 27B and two VLMs included (#16)\n\nThe shape matrix now covers everything downloaded, and the numbers are finally\ntrustworthy enough to publish. int4 group 64, identical weights both sides:\n\n  model             hidden   inter   pre up  pre down   1 row  8 rows  16 rows\n  Qwen2.5 0.5B         896    4864    1.01x     2.32x   1.02x   1.30x    1.47x\n  Qwen2.5 1.5B        1536    8960    1.03x     2.79x   1.02x   1.23x    1.32x\n  Llama 3.2 1B        2048    8192    1.06x     3.09x   0.98x   1.27x    1.81x\n  Llama 3.2 3B        3072    8192    1.07x     1.06x   0.97x   1.23x    1.72x\n  Qwen3.5 4B          2560    9216    1.09x     1.22x   0.97x   1.24x    1.29x\n  Qwen3.5 9B          4096   12288    1.00x     1.15x   1.00x   1.21x    1.27x\n  Qwen3.6 27B         5120   17408    1.09x     1.01x   0.99x   1.21x    1.24x\n  Qwen3-VL 4B         2560    9728    1.10x     1.22x   1.00x   1.22x    1.27x\n  Qwen2.5-VL 7B       3584   18944    1.07x     1.00x   1.01x   1.20x    1.25x\n\nNothing is below parity anywhere. Every model gains 1.20x to 1.30x at eight rows\nand 1.24x to 1.81x at sixteen, and that holds up to 27B and through both vision\nlanguage models. Only prefill varies, and only with width: the three models under\n2560 win it outright, the six at or above it do not.\n\nFor the VLMs only the language tower is measured. The vision encoder runs once\nper image rather than per token, so it is not in the shapes this compares. Note\nqwen2_5_vl exists in mlx_vlm but not mlx_lm, so Qwen2.5-VL-7B has no end-to-end\nfigure, only per-shape.\n\nRounds raised from 15 to 25. At 15, Qwen2.5-1.5B's rows-16 cell came out 1.12x in\none run while three independent measurements of that shape gave 1.31x, 1.31x and\n1.32x. A 15% error in a number someone reads off a table is worth the extra\nminutes, and the earlier 0.75x and 0.79x entries that turned out to be a real\nselection bug were only distinguishable from noise because they were chased.\n\n593 tests pass.\n\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-29T19:06:11-04:00",
          "tree_id": "fbd02c1a62a87abb552a5207a21edbe1848e08cc",
          "url": "https://github.com/AndreSlavescu/meTile/commit/c5a9beea831aacfd9dd62fb21e0391bb7fb6a94a"
        },
        "date": 1785366537737,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 531.72,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 4972.37,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 428.84,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1288.17,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 440.83,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1367.33,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 435.23,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 405.78,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 436.68,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 463.53,
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
          "id": "19d8c11b70d6a8499a2df64817223bb1a07bc3ef",
          "message": "Test that whole models generate the same tokens as MLX (#17)\n\nThe kernel tests check numerics one kernel at a time, and the model plan gate\nchecks logits for a single next-token step. Neither answers the question a user\nactually has, and both can pass while generation diverges.\n\nTwo reasons they can. The quantized compatibility gates are tolerance based at\nrtol 3e-2, which is invisible per layer and compounds across 32 to 64 of them.\nAnd greedy decoding takes an argmax, which is discontinuous: two logit vectors\n1e-3 apart agree almost always and disagree when the top two candidates are\nclose, after which the sequences never reconverge. A single-step logit check\ncannot see either.\n\nSo this generates 48 tokens greedily at temperature 0 and compares token ids\nposition by position, across every small model in the local cache. On divergence\nit reports the index, both tokens and both decoded tails, because diverging at\ntoken 2 and at token 37 are different bugs.\n\nResult: Qwen2.5-0.5B, Qwen2.5-1.5B and Llama-3.2-1B all match MLX token for\ntoken over 48 steps.\n\nTwo things the tests do beyond the comparison, because a green correctness test\nthat has stopped measuring is worse than no test:\n\n_assert_patched checks that the patch context really swaps a layer's bound\nimplementations and restores them on exit. Without it, an API change that made\napply_metile_to_mlx_lm a no-op would leave this suite passing while exercising no\nmeTile kernel at all. Verified by handing it a deliberately empty context\nmanager, which it rejects. On Qwen2.5-0.5B the swap covers mlp and\ninput_layernorm; self_attn is not swapped at layer level, so this test does not\ncover the attention path and does not claim to.\n\nThe second test generates twice under meTile and requires the same output.\nKernel selection is decided by measurement, so a second run can select\ndifferently; if that changed the tokens, comparing against MLX would be measuring\nthe tuner rather than the kernels, and this says which of the two failed.\n\nMarked slow and registered in pyproject, so `-m \"not slow\"` skips them and CI\nstays green without models cached. They skip rather than fail when the cache is\nempty.\n\n597 tests pass; 593 with the slow ones deselected.\n\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-29T16:41:16-07:00",
          "tree_id": "398a0f07b8fce722561eb64d9f7ee9c63c3332c3",
          "url": "https://github.com/AndreSlavescu/meTile/commit/19d8c11b70d6a8499a2df64817223bb1a07bc3ef"
        },
        "date": 1785368644029,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 454.04,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 3811.84,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 389.21,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1142.52,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 338.65,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1198.77,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 329.13,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 285.24,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 339.06,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 385.36,
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
          "id": "4babc544694b4c3706c4d74a247504bdb88b6466",
          "message": "Registry-driven model patch tests, and stop attention crashing on head_dim 256 (#18)\n\nRestructured after the Liger pattern: MODEL_CASES lists checkpoints, FEATURE_SETS\nlists patch surfaces, and the test runs the product. Adding a model or a subsystem\nis one line. The previous version had a bare tuple of three names and no notion of\nwhich subsystem was under test, so a failure said \"this model broke\" and nothing\nmore.\n\nPer-subsystem runs are the point. Each feature set disables everything but one, so\na red cell names a suspect, and the all-on case still catches interactions. What\neach set actually reaches, verified rather than assumed:\n\n  attention      -> mlx_lm.models.*.scaled_dot_product_attention\n  graph_fusion   -> the decoder block's own __call__\n  quantized_mlp  -> block + mlp\n  rms_norm       -> both layernorms\n\nFinding the blocks needed to stop assuming model.model.layers. That path raises\nAttributeError on Qwen3.5, which nests differently, and the block list has to be\nlocated structurally. Recognising linear_attn alongside self_attn is what makes a\nhybrid model visible at all: Qwen3.5 and Qwen3.6 alternate GatedDeltaNet layers\nwith standard attention and use Qwen3NextMLP, and meTile patches neither the MLP\nnor the block there. Since the MLP is where the decode speedup comes from, that,\nnot the width cliff, is the better explanation for those models reporting 1.000x.\n\nTwo bugs the tests found:\n\nAttention crashed instead of falling back. head_dim 256, which Qwen3.5, Qwen3.6\nand Qwen3-VL all use, satisfies every condition the shape gate checks and then\nneeds 40960 bytes of threadgroup memory against a 32768-byte limit, so the kernel\nraised into the caller's generate loop. Shapes that fail to build are now recorded\nand served from MLX. Deriving a head-dimension bound arithmetically was the\nalternative and is worse: it hardcodes the current kernel's allocation formula\ninto the gate and drifts the first time the tiling changes.\n\n_implementation replaces getattr(cls, \"__call__\"). For a class that does not define\n__call__, getattr resolves through the metaclass and returns a fresh method-wrapper\nper access, so the identity check reported a swap that never happened. A false\npositive in that guard is worse than no guard: it claims coverage that is absent.\nCaught by the no-op detector test, which is itself the thing that keeps this suite\nfrom silently measuring nothing.\n\nOne divergence recorded, not hidden. Qwen3-VL-4B with attention diverges from MLX\nat token 7 of 48, and its decode logits differ by 0.43 against a maximum magnitude\nof 27.4. Strict xfail, so it cannot pass quietly and a fix shows up as an\nunexpected pass. Ruled out by measurement: the kernel is clean in isolation across\nfloat16 and bfloat16, head dimensions 64 and 128, grouped-query ratios 4 and 7, and\nevery key count from 1 to 128; the other three subsystems are bit-exact on this\nsame model; both models use the same KVCache class and step. What differs between\nthe real invocation and every synthetic reproduction is still open.\n\nAlso measured, per subsystem at a real decode step rather than prefill (meTile\nattention only engages when the query length is 1, so a prefill comparison never\nruns the kernel): Qwen2.5-0.5B is bit-exact on all four subsystems, and Qwen3-VL\nis bit-exact on three.\n\n610 tests pass, 25 skipped, 2 xfailed; 594 with slow deselected. Vulture clean.\n\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-29T17:10:58-07:00",
          "tree_id": "38cf694c5dcf66118e5cb0cabdb557dab931c2d2",
          "url": "https://github.com/AndreSlavescu/meTile/commit/4babc544694b4c3706c4d74a247504bdb88b6466"
        },
        "date": 1785370459914,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 493.33,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 4454.37,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 390.15,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1493.15,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 419.25,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1479.06,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 354.97,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 347.68,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 359.09,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 468.93,
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
          "id": "c66cf80103a8f6af927bda232ffb2b9738a845e4",
          "message": "Accumulate attention in f32, and assert logit equality instead of token equality (#19)\n\nThe attention decode kernel multiplied two storage-dtype loads together. In MSL\nbfloat * bfloat yields bfloat, so every one of the D dot-product terms rounded to\nan 8-bit significand before it ever reached the f32 accumulator, and the same\nhappened to probability * value. Casting the Q, K and V loads to f32 first fixes\nit.\n\nMeasured against a float32 reference on Qwen3-VL-4B, 36 real decode-step calls:\n\n                       before      after\n  median MLX error     0.003403   0.003403\n  median meTile error  0.013213   0.003403\n  meTile worse than    33/36      0/36\n  max |meTile - MLX|   0.062500   0.000000\n\nAll 36 calls are now bit-exact, and the accuracy is MLX's exactly rather than 4x\nworse. No speed cost: bf16 attention still measures 1.30x at 1024 keys and 1.18x\nat 256, f16 stays at parity.\n\nFinding it took discarding three wrong measurements, each worth naming because\neach looked conclusive. Comparing at prefill reported everything bit-exact, which\nwas true and meaningless: meTile attention only engages when the query length is\n1, so prefill never runs the kernel. Expressing the error in ulps of the tensor's\nmaximum said \"rounding-level\"; per-element ulps said 160000, because a 1e-6 floor\nmakes near-zero elements meaningless. What settled it was comparing both\nimplementations against a float32 reference, where meTile was plainly 4x further\nfrom the truth.\n\nThe tests now assert bit-exact logits rather than identical tokens. Tokens are the\nweaker property: two logit vectors can differ and argmax the same way for many\nsteps, so a token test passes over a real numeric regression and then fails later\non something unrelated. Switching contract immediately surfaced two more\ndivergences the token tests had missed.\n\nBoth turned out to be reduction order where meTile is the more accurate side, so\nthey are bounded and documented rather than eliminated. Measured against float32\nat kernel level: MLX's f16 SwiGLU errs 18.05 from truth against meTile's 4.10 at\nhidden 2048 and inter 8192, and its f16 RMSNorm errs 0.00293 against 0.00185 at\nhidden 3072. Matching bit-for-bit there means adopting a measurably worse\nsummation order. Every pair not listed must be exactly equal, which is what\ncaught this kernel.\n\nThe Qwen3-VL xfail is retired. It reported XPASS(strict) once the cast landed,\nwhich is how that mechanism is meant to announce a fix.\n\n66 pass in the model matrix, 16 skipped; 596 with slow deselected. Vulture clean.\n\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-29T18:07:31-07:00",
          "tree_id": "5a1ab62ea7d439f1b00079780047b5950d65c3a7",
          "url": "https://github.com/AndreSlavescu/meTile/commit/c66cf80103a8f6af927bda232ffb2b9738a845e4"
        },
        "date": 1785373855021,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 470.31,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 4036.42,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 374.65,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1268.71,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 352.56,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1106.3,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 335.01,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 332.25,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 388.7,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 413.26,
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
          "id": "11d3cba4808f774ca9d070708078330f83c1f60b",
          "message": "Schedule instructions natively, and measure that it does nothing (#20)\n\n* Accumulate attention in f32, and assert logit equality instead of token equality\n\nThe attention decode kernel multiplied two storage-dtype loads together. In MSL\nbfloat * bfloat yields bfloat, so every one of the D dot-product terms rounded to\nan 8-bit significand before it ever reached the f32 accumulator, and the same\nhappened to probability * value. Casting the Q, K and V loads to f32 first fixes\nit.\n\nMeasured against a float32 reference on Qwen3-VL-4B, 36 real decode-step calls:\n\n                       before      after\n  median MLX error     0.003403   0.003403\n  median meTile error  0.013213   0.003403\n  meTile worse than    33/36      0/36\n  max |meTile - MLX|   0.062500   0.000000\n\nAll 36 calls are now bit-exact, and the accuracy is MLX's exactly rather than 4x\nworse. No speed cost: bf16 attention still measures 1.30x at 1024 keys and 1.18x\nat 256, f16 stays at parity.\n\nFinding it took discarding three wrong measurements, each worth naming because\neach looked conclusive. Comparing at prefill reported everything bit-exact, which\nwas true and meaningless: meTile attention only engages when the query length is\n1, so prefill never runs the kernel. Expressing the error in ulps of the tensor's\nmaximum said \"rounding-level\"; per-element ulps said 160000, because a 1e-6 floor\nmakes near-zero elements meaningless. What settled it was comparing both\nimplementations against a float32 reference, where meTile was plainly 4x further\nfrom the truth.\n\nThe tests now assert bit-exact logits rather than identical tokens. Tokens are the\nweaker property: two logit vectors can differ and argmax the same way for many\nsteps, so a token test passes over a real numeric regression and then fails later\non something unrelated. Switching contract immediately surfaced two more\ndivergences the token tests had missed.\n\nBoth turned out to be reduction order where meTile is the more accurate side, so\nthey are bounded and documented rather than eliminated. Measured against float32\nat kernel level: MLX's f16 SwiGLU errs 18.05 from truth against meTile's 4.10 at\nhidden 2048 and inter 8192, and its f16 RMSNorm errs 0.00293 against 0.00185 at\nhidden 3072. Matching bit-for-bit there means adopting a measurably worse\nsummation order. Every pair not listed must be exactly equal, which is what\ncaught this kernel.\n\nThe Qwen3-VL xfail is retired. It reported XPASS(strict) once the cast landed,\nwhich is how that mechanism is meant to announce a fix.\n\n66 pass in the model matrix, 16 skipped; 596 with slow deselected. Vulture clean.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* Schedule instructions natively, and measure that it does nothing\n\nA native scheduling pass over Metal IR: a dependence-correct list scheduler whose objective is\nregister pressure rather than latency, plus an opt-in reassociation pass for\ninstruction-level parallelism. Both are off by default, and the reason is the measurement.\n\nPressure first is the right objective for this target and comes straight from our own numbers.\nReaching the register budget measured 1.3x to 6.7x slower; perfect ILP is worth at most 1.09x\non fp32 and 1.41x on fp16. The spill cliff is between one and six times the entire ILP prize,\nso the scheduler chases latency only while pressure is comfortable and switches to relieving\nit past 80% of the budget, which is the standard integrated-prepass shape.\n\nIt does not work, and that is the finding. Across six kernels from 14 to 126 allocated\nregisters, reordering changed the register count in none of them, and no timing difference\nsurvived the benchmark's own control. The control is the part worth keeping: cases where the\npass emits byte-identical MSL are timed too, and identical source cannot beat itself, so their\nspread is the noise floor. It read 0.6% to 7.6% between runs. An early version of the\nbenchmark would have reported a 1.28x win on softmax while the control row beside it read\n0.75x on identical source at the same moment; three repeats gave 1.050x, 0.985x, 0.893x.\n\nThe cause is structural rather than a shortcoming of the pass. Apple's backend schedules and\nallocates from the MSL it is handed, so statement order is a suggestion, and on this evidence\nit is declined. This is the same conclusion the ILP ceiling reached, from a second direction:\nthere is no scheduling win available above MSL. The passes stay because they are correct,\ntested against the hazards that would make them miscompile, and they operate at the level a\nscheduler has to operate at if meTile emits machine code directly, which the binary-archive\ninjection work established is possible.\n\nGetting the dependences right took two fixes, both of which produced kernels Metal rejected\nwith \"use of undeclared identifier\":\n\n  Object identity is the wrong key for a value. The lowering hands out several distinct MValue\n  objects carrying one name, so keying on id() splits one variable into many and loses edges.\n\n  The raw name is also wrong. CSE forwards a redundant value to its equivalent without\n  renaming it, so two names can mean one variable. The emitter already resolved this before\n  printing; the rule now lives in mir.resolve, which both the emitter and the scheduler use,\n  because a pass that disagrees with it silently loses dependence edges.\n\nReassociation is separate and stays off for a reason beyond speed: it reorders\nfloating-point addition, and the model tests now assert bit-exact logits against MLX. Trading\nthat for at most 9% is a bad trade. It also had to be taught to delete the additions it\nabsorbs, since a rebuilt chain that leaves them behind emits a tree and a chain, spending more\ninstructions to shorten a dependence, and to find the maximal chain rather than the first\nqualifying one, since rebuilding four terms of an eight-term chain barely shortens anything.\n\n16 scheduling tests, built around hazards rather than outcomes: what must not move across a\nbarrier, across an operation whose effects the pass does not model, past a variable\nreassignment, or past a store through an aliased pointer. The end-to-end test compares output\nbits with the pass on and off and additionally requires the two runs to have compiled\n*different* MSL, because otherwise agreeing outputs would prove nothing.\n\n610 pass with the pass off, 610 with it forced on. Lint and vulture clean.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-29T18:31:12-07:00",
          "tree_id": "8a9b76c36b29d48281ed5f1b18081d5cd330d740",
          "url": "https://github.com/AndreSlavescu/meTile/commit/11d3cba4808f774ca9d070708078330f83c1f60b"
        },
        "date": 1785375226295,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 372.08,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 3199.59,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 317.35,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 961.02,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 326.74,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1015.1,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 250.2,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 251.03,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 281.4,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 318.84,
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
          "id": "de2a7284adaa69c536c527b25899a746d20fe377",
          "message": "Read back machine code, and prove statement order never reaches it (#21)\n\nThe scheduling pass measured flat, and a flat timing is a weak conclusion: it says the\nharness could not see a difference, not that there is none. Comparing the compiled __text\nsettles it outright, because two source forms that produce identical bytes cannot differ in\nspeed and no measurement is required to say so.\n\n`metile.target.agx.machine_code` returns a kernel's __text, reusing the binary-archive\nunwrapping that already backed the register reader. `benchmarks/agx_source_order.py` asks\nevery code-generation choice a compiler above MSL is in a position to make:\n\n  statement order    two independent fma chains written serially and written interleaved\n                     compile to the same 190 bytes. A load at its use and the same load\n                     hoisted compile to the same 218.\n  reassociation      a serial addition chain and a balanced tree over the same eight terms\n                     compile to the same 282 bytes.\n  live-range shape   eight to sixty-four values held live against consumed as they arrive:\n                     within one register, identical code size at every count.\n\nThree of three normalised away. Apple's backend rebuilds the schedule from the dataflow it\nis handed, so statement order is not an instruction, and nothing meTile can express in MSL\nmoves these bytes.\n\nThat is the real reason metile/compiler/scheduling.py is off, and it is a better reason than\nthe timings in the previous commit. It also retires the reassociation pass's supposed\ntrade-off from the other side: rebuilding a chain into a tree cannot buy the ILP it was\nwritten for, because the backend already emits the same instructions for both. The pass was\noff for costing bit-exactness to gain at most 9%; it turns out to gain nothing.\n\nThe conclusion is about where the boundary of our control sits, not about the passes. Above\nMSL the leverage is which algorithm, which tiling and which functional unit, and those are\nworth 2.4x to 3.7x against scheduling's 1.09x ceiling. Below MSL is where a scheduler would\nbite, and the binary-archive work established that meTile can write there.\n\nThe normalisation property is now a test rather than a note, because it is load bearing: the\npass is disabled on the assumption that it holds. If a toolchain update breaks it the test\nfails loudly and says to reconsider the default, instead of the assumption rotting quietly.\n\n611 pass. Lint and vulture clean.\n\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-29T18:37:22-07:00",
          "tree_id": "eccb112eb961d25d854d1f3aee0d0fd7d4c5eda8",
          "url": "https://github.com/AndreSlavescu/meTile/commit/de2a7284adaa69c536c527b25899a746d20fe377"
        },
        "date": 1785375620272,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 391.83,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 3109.55,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 377.9,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1203.75,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 346.94,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1251.87,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 272.63,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 293.63,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 285.23,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 340.6,
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
          "id": "e49f0fa01974332fc8aaa086a1b65d69458eaccd",
          "message": "Assemble G17 fma instructions from scratch, registers included (#23)\n\nThe register field was the last piece needed to build an instruction rather than edit one. The\nindex appears twice in the compact form, as byte 0's high nibble and as (r << 1) | 1 in byte 1,\nand the two agreed in every instruction examined across three independent fma chains. Confirmed\nthe only way that counts: redirect an instruction onto another chain's register and predict the\nwhole kernel's output. Three redirects, three exact matches.\n\nWith registers, constants and flags all measured, `encode_fma` assembles the form outright. It\nreproduces the compiler's own bytes byte for byte on the cases the compiler emits, which is the\ncheapest available check on an assembler -- agree with the only other one in existence -- and\nthen goes past it. Four forms no Metal compiler produced, each predicted on four inputs before\nthe bytes were assembled and each exact:\n\n    a*3+0.5          87.5, 141.5, 195.5, 303.5\n    a*1.5-2          0.625, 7.375, 14.125, 27.625\n    a*7, no addend   1029, 1715, 2401, 3773\n    -a*2+1           -21, -37, -53, -85\n\n`a*1.5-2` is worth noting: the immediate field is unsigned, so a negative addend is not\nrepresentable in it at all and the sign has to travel in the control byte's negate bit. The\nencoder does that itself.\n\nOne prediction failed first, and it is the reason to write predictions down in advance. A\nsynthesised `a*7` measured 1536 from x=1 against 1029 predicted -- eight-fold growth per step\nwhere seven was asked for -- because dropping the addend wrote 0x00 into its immediate slot.\nZero there is not inert: it selects a register operand, register 0 happened to be the\naccumulator, and the kernel computed a*7 + a. The slot now keeps an ordinary encoded constant\nand the control bit alone disables it, which is the configuration the flag scan had verified.\nNothing about that would have been visible from reading the bytes.\n\nThe register field also explains an earlier miss. A redirect predicted 976 and measured 980, and\nthe field was right: chain a had three fmas but only two in the compact form, the one consuming a\nfreshly loaded value using a longer encoding. The replay now derives the hidden count per chain\nand the baseline check is what licenses it -- 488 predicted, 488 measured.\n\n26 ISA tests, 637 in total. The probe re-derives six stages and exits non-zero if any prediction\nmisses. Lint and vulture clean.\n\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-29T19:05:26-07:00",
          "tree_id": "0d12872b2da6b77bb903e54af131fde57a1d221e",
          "url": "https://github.com/AndreSlavescu/meTile/commit/e49f0fa01974332fc8aaa086a1b65d69458eaccd"
        },
        "date": 1785377293640,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 457.37,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 4121.66,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 479.41,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1364.21,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 506.8,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1371.85,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 425.86,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 447.25,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 369.59,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 432.42,
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
          "id": "91ed1411aa313128f23c03585f1445026cbac22d",
          "message": "Correct the addend flag: it picks immediate or register, not present or absent (#25)\n\nA flag landed under the wrong name, and the wrong name came from a reading that had already\nsurvived a prediction, which is the interesting part. It was ADDEND_ENABLE, described as\nincluding the addend or not, because clearing it turned a*m+d into a*m on four inputs.\n\nClearing it actually switches the addend slot from an immediate to a register. The synthesised\ninstruction that verified \"a*m\" had an ordinary encoded constant left in the slot, 0xb0, and with\nthe immediate bit clear that byte names register 88 -- above the sixteen the field reaches, so it\nreads zero. The addend was never absent. It was zero, by accident, and the prediction passed\nbecause zero and absent are indistinguishable in a sum.\n\nThe slot uses the same shape as the register field elsewhere, `r << 1` with the low bit ignored.\nVerified by rewriting instructions to `rd = rd * m + rs` for every ordered pair of three live\nregisters at two multipliers, predicting all four threads each: eighteen rewrites, seventy-two\nexact values.\n\nThat gives the register-plus-register add, as `rd * 1 + rs`, which no immediate form can express.\n`encode_fma` now takes either an immediate addend or an `addend_register` and refuses both at\nonce, and a zero addend is stated for what it is: a register index chosen because it cannot be\nreached, with a test asserting that property so a future part with more registers cannot silently\nturn it into whatever that register holds.\n\nOne more prediction miss worth recording, because the encoding was right and my model was wrong.\nThe register-addend rewrites first matched on thread 0 and failed on the other three: thread gid\nreads x[gid], x[gid+1] and x[gid+2], and the prediction used thread 0's inputs throughout. 18 of\n18 once the shift was applied. When a hypothesis matches exactly on one thread and misses on the\nrest, the harness is the suspect, not the field map.\n\n29 ISA tests. Lint clean.\n\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-07-29T19:53:55-07:00",
          "tree_id": "118948839a110b1d47e3a1de65949804022196fe",
          "url": "https://github.com/AndreSlavescu/meTile/commit/91ed1411aa313128f23c03585f1445026cbac22d"
        },
        "date": 1785380238126,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "gemm_256x256x256",
            "value": 484.96,
            "unit": "us"
          },
          {
            "name": "gemm_1024x1024x1024",
            "value": 4050.42,
            "unit": "us"
          },
          {
            "name": "softmax_256x1024",
            "value": 376,
            "unit": "us"
          },
          {
            "name": "softmax_1024x4096",
            "value": 1214.59,
            "unit": "us"
          },
          {
            "name": "layernorm_256x1024",
            "value": 381.78,
            "unit": "us"
          },
          {
            "name": "layernorm_1024x4096",
            "value": 1203.03,
            "unit": "us"
          },
          {
            "name": "fft_1x256",
            "value": 382.85,
            "unit": "us"
          },
          {
            "name": "fft_32x256",
            "value": 341.86,
            "unit": "us"
          },
          {
            "name": "fft_1x1024",
            "value": 359.14,
            "unit": "us"
          },
          {
            "name": "fft_128x1024",
            "value": 424.58,
            "unit": "us"
          }
        ]
      }
    ]
  }
}