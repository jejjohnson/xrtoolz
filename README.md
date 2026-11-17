# xrtoolz has moved to geotoolz

xrtoolz's three packages, their history and their open issues now live in [geotoolz](https://github.com/jejjohnson/geotoolz), and this repository is read-only.

| Package | Import | In geotoolz |
|---|---|---|
| `xrtoolz` | `xrtoolz` | [`packages/xrtoolz`](https://github.com/jejjohnson/geotoolz/tree/main/packages/xrtoolz) |
| `xrtoolz-ops` | `xrops` | [`packages/xrtoolz-ops`](https://github.com/jejjohnson/geotoolz/tree/main/packages/xrtoolz-ops) |
| `xrtoolz-products` | `xrproducts` | [`packages/xrtoolz-products`](https://github.com/jejjohnson/geotoolz/tree/main/packages/xrtoolz-products) |

The documentation is the [xarray lane section](https://jejjohnson.github.io/geotoolz/xarray/) of the geotoolz site. Releases continue from geotoolz under the same tags (`xrtoolz-ops-vX.Y.Z`, …), and issues are filed on [geotoolz](https://github.com/jejjohnson/geotoolz/issues). Old issue links redirect there.

Install from geotoolz:

```bash
uv pip install "xrtoolz-ops @ git+https://github.com/jejjohnson/geotoolz@main#subdirectory=packages/xrtoolz-ops"
```
