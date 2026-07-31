import customtkinter as ctk


class ConsoleWidget(
    ctk.CTkTextbox
):

    def __init__(
        self,
        parent
    ):

        super().__init__(
            parent
        )

    def write(
        self,
        text
    ):

        self.insert(
            "end",
            str(text) + "\n"
        )

        self.see(
            "end"
        )

    def clear(
        self
    ):

        self.delete(
            "1.0",
            "end"
        )