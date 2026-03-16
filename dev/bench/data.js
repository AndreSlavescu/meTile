window.BENCHMARK_DATA = {
  "lastUpdate": 1773705015461,
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
      }
    ]
  }
}