# Faster Whisper on Beam

When deployed, this can be invoked with either a URL to an .mp3 file or a base64-encoded audio file

In your shell, serve this by running:

`beam serve app.py:transcribe`

Then, the API can be invoked like this:

```bash
 curl -X POST 'https://app.beam.cloud/endpoint/id/[YOUR-ENDPOINT-ID]' \
-H 'Connection: keep-alive' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer [YOUR-AUTH-TOKEN]' \
-d '{"url":"http://commondatastorage.googleapis.com/codeskulptor-demos/DDR_assets/Kangaroo_MusiQue_-_The_Neverwritten_Role_Playing_Game.mp3"}'
```
