from beam import Output, endpoint, Image, env
import asyncio

if env.is_remote():
    from playwright.async_api import async_playwright

image = (
    Image(python_version="python3.11")
    .add_python_packages(
        [
            "playwright",
        ]
    )
    .add_commands(["playwright install chromium", "playwright install-deps chromium"])
)


@endpoint(name="headless-browser", cpu=2, memory="16Gi", image=image)
async def browser(url: str = "https://example.com"):
    print(f"Navigating to: {url}")
    output_path = "/tmp/screenshot.png"

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        print("Browser launched successfully")

        try:
            page = await browser.new_page()
            await page.set_viewport_size({"width": 1920, "height": 1080})

            await page.goto(url, wait_until="networkidle")
            print("Page loaded, waiting 2 seconds for any dynamic content...")

            await asyncio.sleep(2)

            await page.screenshot(path=output_path, full_page=True, type="png")
            print(f"Screenshot saved as: {output_path}")

        finally:
            await browser.close()

    output_file = Output(path=output_path)
    output_file.save()
    public_url = output_file.public_url(expires=400)

    return {"output_url": public_url}
