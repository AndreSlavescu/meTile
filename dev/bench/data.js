window.BENCHMARK_DATA = {
  "lastUpdate": 1773992702996,
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
      }
    ]
  }
}