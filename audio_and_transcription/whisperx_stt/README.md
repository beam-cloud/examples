# WhisperX Inference

## Deploy the endpoint in `app.py`

```
beam deploy app.py:transcribe_audio
```

## Send a request to the API

Add the deployment URL, auth token, and a URL with your audio file to `request.py`:

```
AUTH_TOKEN = "BEAM_AUTH_TOKEN"
BEAM_URL = "id/8836f704-b521-4e1c-8979-bc74c97dc47b"
AUDIO_URL = ""
```

> [You can find various audio samples here.](https://audio-samples.github.io/samples/mp3/blizzard_unconditional/sample-0.mp3)

Send the request by running `python request.py`. You'll get back a response like this:

```
{"result":{"segments":[{"start":0.309,"end":3.133,"text":" My thought, I have nobody by a beauty and will as you t'ward.","words":[{"word":"My","start":0.309,"end":0.45,"score":0.93},{"word":"thought,","start":0.49,"end":0.85,"score":0.863},{"word":"I","start":0.91,"end":0.97,"score":0.999},{"word":"have","start":1.01,"end":1.191,"score":0.874},{"word":"nobody","start":1.251,"end":1.571,"score":0.91},{"word":"by","start":1.611,"end":1.751,"score":0.863},{"word":"a","start":1.791,"end":1.832,"score":0.975},{"word":"beauty","start":1.872,"end":2.152,"score":0.836},{"word":"and","start":2.192,"end":2.272,"score":0.82},{"word":"will","start":2.292,"end":2.472,"score":0.853},{"word":"as","start":2.512,"end":2.593,"score":0.838},{"word":"you","start":2.613,"end":2.753,"score":0.842},{"word":"t'ward.","start":2.793,"end":3.133,"score":0.217}]},{"start":3.874,"end":9.943,"text":"Mr. Rochester is sub, and that so don't find simpus, and devoted abode, to hath might in a","words":[{"word":"Mr.","start":3.874,"end":4.175,"score":0.563},{"word":"Rochester","start":4.235,"end":4.756,"score":0.94},{"word":"is","start":4.836,"end":4.916,"score":0.816},{"word":"sub,","start":4.936,"end":5.236,"score":0.877},{"word":"and","start":5.276,"end":5.356,"score":0.802},{"word":"that","start":5.397,"end":5.577,"score":0.948},{"word":"so","start":5.617,"end":5.777,"score":0.982},{"word":"don't","start":5.817,"end":6.017,"score":0.863},{"word":"find","start":6.057,"end":6.358,"score":0.873},{"word":"simpus,","start":6.398,"end":6.839,"score":0.865},{"word":"and","start":7.399,"end":7.499,"score":0.884},{"word":"devoted","start":7.54,"end":7.92,"score":0.969},{"word":"abode,","start":8.0,"end":8.461,"score":0.635},{"word":"to","start":9.102,"end":9.222,"score":0.839},{"word":"hath","start":9.262,"end":9.402,"score":0.65},{"word":"might","start":9.442,"end":9.703,"score":0.855},{"word":"in","start":9.783,"end":9.883,"score":0.8},{"word":"a","start":9.923,"end":9.943,"score":0.97}]}],"word_segments":[{"word":"My","start":0.309,"end":0.45,"score":0.93},{"word":"thought,","start":0.49,"end":0.85,"score":0.863},{"word":"I","start":0.91,"end":0.97,"score":0.999},{"word":"have","start":1.01,"end":1.191,"score":0.874},{"word":"nobody","start":1.251,"end":1.571,"score":0.91},{"word":"by","start":1.611,"end":1.751,"score":0.863},{"word":"a","start":1.791,"end":1.832,"score":0.975},{"word":"beauty","start":1.872,"end":2.152,"score":0.836},{"word":"and","start":2.192,"end":2.272,"score":0.82},{"word":"will","start":2.292,"end":2.472,"score":0.853},{"word":"as","start":2.512,"end":2.593,"score":0.838},{"word":"you","start":2.613,"end":2.753,"score":0.842},{"word":"t'ward.","start":2.793,"end":3.133,"score":0.217},{"word":"Mr.","start":3.874,"end":4.175,"score":0.563},{"word":"Rochester","start":4.235,"end":4.756,"score":0.94},{"word":"is","start":4.836,"end":4.916,"score":0.816},{"word":"sub,","start":4.936,"end":5.236,"score":0.877},{"word":"and","start":5.276,"end":5.356,"score":0.802},{"word":"that","start":5.397,"end":5.577,"score":0.948},{"word":"so","start":5.617,"end":5.777,"score":0.982},{"word":"don't","start":5.817,"end":6.017,"score":0.863},{"word":"find","start":6.057,"end":6.358,"score":0.873},{"word":"simpus,","start":6.398,"end":6.839,"score":0.865},{"word":"and","start":7.399,"end":7.499,"score":0.884},{"word":"devoted","start":7.54,"end":7.92,"score":0.969},{"word":"abode,","start":8.0,"end":8.461,"score":0.635},{"word":"to","start":9.102,"end":9.222,"score":0.839},{"word":"hath","start":9.262,"end":9.402,"score":0.65},{"word":"might","start":9.442,"end":9.703,"score":0.855},{"word":"in","start":9.783,"end":9.883,"score":0.8},{"word":"a","start":9.923,"end":9.943,"score":0.97}]}}
```
