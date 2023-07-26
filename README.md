#### hosting local file server
https://github.com/danvk/RangeHTTPServer

```
$ pip install rangehttpserver
$ python -m RangeHTTPServer 8070
Serving HTTP on 0.0.0.0 port 8000 ...
```

#### Uvicorn titiler tile server
`uvicorn titiler.application.main:app --reload --port 8090`

#### Creating tiled geotiff that can be read as Deck.gl `TileLayer`
`python create_geotiff_pyramid.py <input_path> <output_path>`

```
python create_geotiff_pyramid.py /home/GEMINI/Dataset/Dev_Test/7-25-2022-orthophoto-Davis.tif /home/GEMINI/Dataset/Dev_Test/7-25-2022-orthophoto-Davis-Pyramid.tif
```

#### Remote port forwarding
`$ssh -L 3031:localhost:3031 <YOUR_USER>@<HOST_IP>`

Make sure that this matches your server started by `npm start`