# Headless Browser API Example

A headless browser API powered by Playwright, capturing full-page website screenshots.

## Deployment

Deploy the app on Beam:

```
beam deploy app.py:browser
```

## API Usage

Send a `POST` request to the endpoint with the following JSON body:

```json
{
  "url": "https://your-website-url"
}
```

### Example Request:

```json
{
  "url": "https://example.com"
}
```

### Example Response:

```json
{
  "output_url": "https://app.beam.cloud/output/id/9dfbb7a1-a3de-489c-a602-423b4c859f84"
}
```