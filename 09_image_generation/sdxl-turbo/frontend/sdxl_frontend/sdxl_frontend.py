import reflex as rx
import sdxl_frontend.beam_service as client


class State(rx.State):
    """The app state."""

    prompt = ""
    image_url = ""
    processing = False
    complete = False

    def get_image(self):
        """Get the image from the prompt."""
        if not self.prompt:
            return rx.window_alert("Prompt Empty")

        self.processing, self.complete = True, False

        beam_client = client.BeamService(prompt=self.prompt)
        response = beam_client.call_api()

        # client.BeamTaskStatus().call_api(response)
        self.image_url = response.get("image", "")
        self.processing, self.complete = False, True


def index():
    return rx.chakra.center(
        rx.chakra.vstack(
            rx.chakra.heading("sdxl turbo on beam.cloud", font_size="1.5em"),
            rx.chakra.input(
                placeholder="Enter a prompt...",
                on_blur=State.set_prompt,
                width="25em",
            ),
            rx.chakra.button(
                "Generate Image",
                on_click=State.get_image,
                width="25em",
                loading=State.processing,
            ),
            rx.chakra.image(src=State.image_url, width="20em"),
            align="center",
        ),
        width="100%",
        height="100vh",
    )


# Add state and page to the app.
app = rx.App()
app.add_page(index, title="sdxl-turbo:beam")
