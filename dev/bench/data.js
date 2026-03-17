window.BENCHMARK_DATA = {
  "lastUpdate": 1773706311770,
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
      }
    ]
  }
}