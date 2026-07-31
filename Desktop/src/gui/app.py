import customtkinter as ctk
import threading

from src.core.paths import (
    get_project_status
)


def dataset_row_count(name: str):
    """Return number of rows for a dataset by name.

    Placeholder implementation to avoid undefined reference in the GUI.
    If more accurate counts are needed, replace with real implementation.
    """
    try:
        status = get_project_status()
        # If status contains counts, try to return them, otherwise unknown -> 0
        datasets_info = status.get("dataset_rows") or {}
        return datasets_info.get(name, 0)
    except Exception:
        return 0

from src.core.paths import (
    get_project_status,
    list_benchmarks,
    list_confusion_matrices,
)

from src.core.settings import (
    SETTINGS
)

from src.pipeline import (

    # =====================================
    # Feature Extraction
    # =====================================

    generate_mfcc,
    generate_demon_lofar,

    # =====================================
    # Dataset Builders
    # =====================================

    build_mfcc_dataset,
    build_demon_dataset,
    build_lofar_dataset,

    # =====================================
    # Training
    # =====================================

    train_classical,
    train_gmm,
    train_hmm,

    train_cnn,
    train_mobilenet,
    train_resnet,

    train_demon_cnn,
    train_lofar_vit,
    train_capse_vit,
    train_catfish,

    # =====================================
    # Pipelines
    # =====================================

    standard_pipeline,
    demon_pipeline,
    lofar_pipeline,
)

from src.gui.widgets.console import (
    ConsoleWidget
)

from src.gui.theme import (

    setup_theme,

    APP_TITLE,

    WINDOW_WIDTH,

    WINDOW_HEIGHT,

    SIDEBAR_WIDTH,
)

from src.core.registry import MODELS

from src.pipeline import (
    train_classical,
    train_gmm,
    train_hmm,

    train_cnn,
    train_mobilenet,
    train_resnet,

    train_demon_cnn,
    train_lofar_vit,
    train_capse_vit,
    train_catfish,

    standard_pipeline,
    demon_pipeline,
    lofar_pipeline,
)



# ============================================================
# Pages
# ============================================================

class DashboardPage(
    ctk.CTkFrame
):

    def __init__(
        self,
        parent
    ):

        super().__init__(
            parent
        )

        self.build_ui()

    def refresh(
        self
    ):

        for widget in self.winfo_children():

            widget.destroy()

        self.build_ui()

    def build_ui(
        self
    ):

        status = get_project_status()

        title = ctk.CTkLabel(

            self,

            text="Project Dashboard",

            font=(
                "Arial",
                30,
                "bold"
            )
        )

        title.pack(
            pady=20
        )
        ctk.CTkButton(

            self,

            text="Refresh Dashboard",

            command=self.refresh

        ).pack(
            pady=10
        )

        feature_frame = ctk.CTkFrame(
            self
        )

        feature_frame.pack(

            fill="x",

            padx=20,

            pady=10
        )

        ctk.CTkLabel(

            feature_frame,

            text="Features",

            font=(
                "Arial",
                20,
                "bold"
            )

        ).pack(
            pady=10
        )

        for feature, count in status[
            "features"
        ].items():

            ctk.CTkLabel(

                feature_frame,

                text=f"{feature.upper()} : {count}"

            ).pack(
                anchor="w",
                padx=20
            )

        dataset_frame = ctk.CTkFrame(
            self
        )

        dataset_frame.pack(

            fill="x",

            padx=20,

            pady=10
        )

        ctk.CTkLabel(

            dataset_frame,

            text="Datasets",

            font=(
                "Arial",
                20,
                "bold"
            )

        ).pack(
            pady=10
        )

        for name, exists in status[
            "datasets"
        ].items():

            state = (
                "✅"
                if exists
                else "❌"
            )

            rows = dataset_row_count(
                name
            )

            ctk.CTkLabel(

                dataset_frame,

                text=f"{name} {state} ({rows} rows)"

            ).pack(
                anchor="w",
                padx=20
            )

        results_frame = ctk.CTkFrame(
            self
        )

        results_frame.pack(

            fill="x",

            padx=20,

            pady=10
        )

        ctk.CTkLabel(

            results_frame,

            text="Results",

            font=(
                "Arial",
                20,
                "bold"
            )

        ).pack(
            pady=10
        )

        ctk.CTkLabel(

            results_frame,

            text=f"Benchmark Files : {status['benchmark_files']}"

        ).pack(
            anchor="w",
            padx=20
        )

        ctk.CTkLabel(

            results_frame,

            text=f"Confusion Matrices : {status['confusion_matrices']}"

        ).pack(
            anchor="w",
            padx=20
        )

        model_frame = ctk.CTkFrame(
            self
        )

        model_frame.pack(

            fill="x",

            padx=20,

            pady=10
        )

        ctk.CTkLabel(

            model_frame,

            text="Models",

            font=(
                "Arial",
                20,
                "bold"
            )

        ).pack(
            pady=10
        )

        ctk.CTkLabel(

            model_frame,

            text=f"Trained Models : {status['trained_models']}"

        ).pack(
            anchor="w",
            padx=20
        )

class DatasetPage(
    ctk.CTkFrame
):

    def __init__(
        self,
        parent
    ):

        super().__init__(
            parent
        )

        self.build_ui()

    def write_console(
        self,
        text
    ):

        self.console.write(
            text
        )

    def start_thread(
        self,
        target
    ):

        thread = threading.Thread(

            target=target,

            daemon=True
        )

        thread.start()

    # =====================================
    # Feature Extraction
    # =====================================

    def run_mfcc(self):

        result = generate_mfcc()

        self.write_console(
            result.message
        )

        if result.stdout:

            self.write_console(
                result.stdout
            )

    def run_demon_lofar(self):

        result = generate_demon_lofar()

        self.write_console(
            result.message
        )

        if result.stdout:

            self.write_console(
                result.stdout
            )

    # =====================================
    # Datasets
    # =====================================

    def run_build_mfcc(self):

        result = build_mfcc_dataset()

        self.write_console(
            result.message
        )

    def run_build_demon(self):

        result = build_demon_dataset()

        self.write_console(
            result.message
        )

    def run_build_lofar(self):

        result = build_lofar_dataset()

        self.write_console(
            result.message
        )

    # =====================================
    # UI
    # =====================================

    def build_ui(
        self
    ):

        title = ctk.CTkLabel(

            self,

            text="Dataset Builder",

            font=(
                "Arial",
                30,
                "bold"
            )
        )

        title.pack(
            pady=20
        )

        # ---------------------------------
        # Feature Extraction
        # ---------------------------------

        feature_frame = ctk.CTkFrame(
            self
        )

        feature_frame.pack(

            fill="x",

            padx=20,

            pady=10
        )

        ctk.CTkLabel(

            feature_frame,

            text="Feature Extraction",

            font=(
                "Arial",
                20,
                "bold"
            )

        ).pack(
            pady=10
        )

        ctk.CTkButton(

            feature_frame,

            text="Generate MFCC",

            command=lambda:
                self.start_thread(
                    self.run_mfcc
                )

        ).pack(
            padx=20,
            pady=5
        )

        status = get_project_status()

        # Models frame
        model_frame = ctk.CTkFrame(
            self
        )

        model_frame.pack(
            fill="x",
            padx=20,
            pady=5
        )

        ctk.CTkLabel(

            model_frame,

            text=f"Trained Models : {status['trained_models']}"

        ).pack(
            anchor="w",
            padx=20
        )

        for model_id, model_info in MODELS.items():

            ctk.CTkLabel(

                model_frame,

                text=model_info["name"]

            ).pack(
                anchor="w",
                padx=40
            )

        ctk.CTkButton(

            feature_frame,

            text="Generate DEMON + LOFAR",

            command=lambda:
                self.start_thread(
                    self.run_demon_lofar
                )

        ).pack(
            padx=20,
            pady=5
        )

        # ---------------------------------
        # Dataset Building
        # ---------------------------------

        dataset_frame = ctk.CTkFrame(
            self
        )

        dataset_frame.pack(

            fill="x",

            padx=20,

            pady=10
        )

        ctk.CTkLabel(

            dataset_frame,

            text="Dataset Generation",

            font=(
                "Arial",
                20,
                "bold"
            )

        ).pack(
            pady=10
        )

        ctk.CTkButton(

            dataset_frame,

            text="Build MFCC Dataset",

            command=lambda:
                self.start_thread(
                    self.run_build_mfcc
                )

        ).pack(
            padx=20,
            pady=5
        )

        ctk.CTkButton(

            dataset_frame,

            text="Build DEMON Dataset",

            command=lambda:
                self.start_thread(
                    self.run_build_demon
                )

        ).pack(
            padx=20,
            pady=5
        )

        ctk.CTkButton(

            dataset_frame,

            text="Build LOFAR Dataset",

            command=lambda:
                self.start_thread(
                    self.run_build_lofar
                )

        ).pack(
            padx=20,
            pady=5
        )

        # ---------------------------------
        # Console
        # ---------------------------------

        buttons = ctk.CTkFrame(
            self
        )

        buttons.pack(

            fill="x",

            padx=20,

            pady=5
        )

        ctk.CTkButton(

            buttons,

            text="Clear Console",

            command=lambda:
                self.console.clear()

        ).pack(
            side="left",
            padx=10
        )

        self.console = ConsoleWidget(
            self
        )

        self.console.pack(

            fill="both",

            expand=True,

            padx=20,

            pady=20
        )

MODEL_FUNCTIONS = {

    "rf": train_classical,

    "xgb": train_classical,

    "gmm": train_gmm,

    "hmm": train_hmm,

    "cnn": train_cnn,

    "mobilenet": train_mobilenet,

    "resnet": train_resnet,

    "demon_cnn": train_demon_cnn,

    "lofar_vit": train_lofar_vit,

    "capse_vit": train_capse_vit,

    "catfish": train_catfish,
}

class TrainingPage(
    ctk.CTkFrame
):

    def __init__(
        self,
        parent
    ):

        super().__init__(
            parent
        )

        self.model_vars = {}

        self.build_ui()
    

    def write_console(
        self,
        text
    ):

        self.console.write(
            text
        )

    def start_thread(
    self,
    target
    ):

        thread = threading.Thread(

            target=target,

            daemon=True
        )

        thread.start()

    def train_selected(
        self
    ):

        self.write_console(
            "Starting Training..."
        )

        for model_id, var in self.model_vars.items():

            if not var.get():
                continue

            self.write_console(
                f"Training {model_id}"
            )

            func = MODEL_FUNCTIONS.get(
                model_id
            )

            if not func:
                continue

            result = func()

            self.write_console(
                f"\n[{result.task_name}]"
            )

            self.write_console(
                result.message
            )

            if result.stdout:

                self.write_console(
                    result.stdout
                )

            if result.stderr:

                self.write_console(
                    "\nERROR:"
                )

                self.write_console(
                    result.stderr
                )

    def run_pipeline(
        self
    ):

        selected = (
            self.pipeline_var.get()
        )

        if selected == "standard":

            results = (
                standard_pipeline()
            )

        elif selected == "demon":

            results = (
                demon_pipeline()
            )

        else:

            results = (
                lofar_pipeline()
            )

        for result in results:

            self.write_console(
                result.message
            )
    def build_ui(
        self
    ):

        title = ctk.CTkLabel(

            self,

            text="Model Training",

            font=(
                "Arial",
                30,
                "bold"
            )

        )

        title.pack(
            pady=20
        )

        top_frame = ctk.CTkFrame(
            self
        )

        top_frame.pack(

            fill="x",

            padx=20,

            pady=10
        )

        ctk.CTkLabel(

            top_frame,

            text="Select Models",

            font=(
                "Arial",
                20,
                "bold"
            )

        ).pack(
            pady=10
        )

        for model_id, model_info in MODELS.items():

            var = ctk.BooleanVar(
                value=False
            )

            self.model_vars[
                model_id
            ] = var

            checkbox = ctk.CTkCheckBox(

                top_frame,

                text=model_info[
                    "name"
                ],

                variable=var
            )

            checkbox.pack(
                anchor="w",
                padx=20
            )

        pipeline_frame = ctk.CTkFrame(
            self
        )

        pipeline_frame.pack(

            fill="x",

            padx=20,

            pady=10
        )

        ctk.CTkLabel(

            pipeline_frame,

            text="Pipeline",

            font=(
                "Arial",
                20,
                "bold"
            )

        ).pack(
            pady=10
        )

        self.pipeline_var = ctk.StringVar(
            value="standard"
        )

        ctk.CTkRadioButton(

            pipeline_frame,

            text="Standard",

            variable=self.pipeline_var,

            value="standard"

        ).pack(
            anchor="w",
            padx=20
        )

        ctk.CTkRadioButton(

            pipeline_frame,

            text="DEMON",

            variable=self.pipeline_var,

            value="demon"

        ).pack(
            anchor="w",
            padx=20
        )

        ctk.CTkRadioButton(

            pipeline_frame,

            text="LOFAR",

            variable=self.pipeline_var,

            value="lofar"

        ).pack(
            anchor="w",
            padx=20
        )

        button_frame = ctk.CTkFrame(
            self
        )

        button_frame.pack(

            fill="x",

            padx=20,

            pady=10
        )

        ctk.CTkButton(

            button_frame,

            text="Train Selected",

            command=lambda:
                self.start_thread(
                    self.train_selected
                )

        ).pack(
            side="left",
            padx=10
        )

        ctk.CTkButton(

            button_frame,

            text="Run Pipeline",

            command=lambda:
                self.start_thread(
                    self.run_pipeline
                )
        ).pack(
            side="left",
            padx=10
        )

        ctk.CTkButton(

            button_frame,

            text="Clear Console",

            command=lambda:
                self.console.clear()

        ).pack(
            side="left",
            padx=10
        )



        self.console = ConsoleWidget(
            self
        )

        self.console.pack(

            fill="both",

            expand=True,

            padx=20,

            pady=20
        )


class ResultsPage(
    ctk.CTkFrame
):

    def __init__(
        self,
        parent
    ):

        super().__init__(
            parent
        )

        self.build_ui()

    def build_ui(
        self
    ):

        title = ctk.CTkLabel(

            self,

            text="Results",

            font=(
                "Arial",
                30,
                "bold"
            )
        )

        title.pack(
            pady=20
        )

        # ======================================
        # Benchmarks
        # ======================================

        benchmark_frame = ctk.CTkFrame(
            self
        )

        benchmark_frame.pack(

            fill="x",

            padx=20,

            pady=10
        )

        ctk.CTkLabel(

            benchmark_frame,

            text="Benchmark Files",

            font=(
                "Arial",
                20,
                "bold"
            )

        ).pack(
            pady=10
        )

        benchmark_files = (
            list_benchmarks()
        )

        if benchmark_files:

            for file in benchmark_files:

                ctk.CTkLabel(

                    benchmark_frame,

                    text=file.name

                ).pack(

                    anchor="w",

                    padx=20
                )

        else:

            ctk.CTkLabel(

                benchmark_frame,

                text="No benchmark files found."

            ).pack(
                padx=20,
                pady=10
            )

        # ======================================
        # Confusion Matrices
        # ======================================

        confusion_frame = ctk.CTkFrame(
            self
        )

        confusion_frame.pack(

            fill="x",

            padx=20,

            pady=10
        )

        ctk.CTkLabel(

            confusion_frame,

            text="Confusion Matrices",

            font=(
                "Arial",
                20,
                "bold"
            )

        ).pack(
            pady=10
        )

        confusion_files = (
            list_confusion_matrices()
        )

        if confusion_files:

            for file in confusion_files:

                ctk.CTkLabel(

                    confusion_frame,

                    text=file.name

                ).pack(

                    anchor="w",

                    padx=20
                )

        else:

            ctk.CTkLabel(

                confusion_frame,

                text="No confusion matrices found."

            ).pack(
                padx=20,
                pady=10
            )

        # ======================================
        # Summary
        # ======================================

        summary_frame = ctk.CTkFrame(
            self
        )

        summary_frame.pack(

            fill="x",

            padx=20,

            pady=10
        )

        ctk.CTkLabel(

            summary_frame,

            text="Summary",

            font=(
                "Arial",
                20,
                "bold"
            )

        ).pack(
            pady=10
        )

        ctk.CTkLabel(

            summary_frame,

            text=f"Benchmark Files: {len(benchmark_files)}"

        ).pack(
            anchor="w",
            padx=20
        )

        ctk.CTkLabel(

            summary_frame,

            text=f"Confusion Matrices: {len(confusion_files)}"

        ).pack(
            anchor="w",
            padx=20
        )


class DeploymentPage(
    ctk.CTkFrame
):
    def __init__(
        self,
        parent
    ):
        super().__init__(parent)

        label = ctk.CTkLabel(

            self,

            text="Deployment",

            font=(
                "Arial",
                28,
                "bold"
            )
        )

        label.pack(
            pady=40
        )

class SettingsPage(
    ctk.CTkFrame
):

    def __init__(
        self,
        parent
    ):

        super().__init__(
            parent
        )

        self.settings = SETTINGS

        self.entries = {}

        self.build_ui()

    def save(
        self
    ):

        settings = {}

        for key, entry in self.entries.items():

            value = entry.get()

            try:

                if "." in value:

                    value = float(
                        value
                    )

                else:

                    value = int(
                        value
                    )

            except ValueError:
                pass

            settings[key] = value

        save_settings(
            settings
        )

    def build_ui(
        self
    ):

        title = ctk.CTkLabel(

            self,

            text="Settings",

            font=(
                "Arial",
                30,
                "bold"
            )
        )

        title.pack(
            pady=20
        )

        form = ctk.CTkScrollableFrame(
            self
        )

        form.pack(

            fill="both",

            expand=True,

            padx=20,

            pady=20
        )

        for key, value in self.settings.items():

            row = ctk.CTkFrame(
                form
            )

            row.pack(

                fill="x",

                pady=5
            )

            ctk.CTkLabel(

                row,

                text=key,

                width=200

            ).pack(
                side="left",
                padx=10
            )

            entry = ctk.CTkEntry(
                row
            )

            entry.insert(
                0,
                str(value)
            )

            entry.pack(

                side="left",

                fill="x",

                expand=True
            )

            self.entries[
                key
            ] = entry

        ctk.CTkButton(

            self,

            text="Save Settings",

            command=self.save

        ).pack(
            pady=20
        )

# ============================================================
# Main Application
# ============================================================

class MESHApp(
    ctk.CTk
):

    def __init__(self):

        super().__init__()

        self.title(
            APP_TITLE
        )

        self.geometry(
            f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"
        )

        self.minsize(
            1200,
            800
        )

        self.create_layout()

    # ========================================================
    # Layout
    # ========================================================

    def create_layout(
        self
    ):

        self.grid_columnconfigure(
            1,
            weight=1
        )

        self.grid_rowconfigure(
            0,
            weight=1
        )

        # ----------------------------------------------------
        # Sidebar
        # ----------------------------------------------------

        self.sidebar = ctk.CTkFrame(

            self,

            width=SIDEBAR_WIDTH,

            corner_radius=0
        )

        self.sidebar.grid(

            row=0,

            column=0,

            sticky="nswe"
        )

        self.sidebar.grid_propagate(
            False
        )

        title = ctk.CTkLabel(

            self.sidebar,

            text="MESH",

            font=(
                "Arial",
                24,
                "bold"
            )
        )

        title.pack(
            pady=20
        )

        self.btn_dashboard = ctk.CTkButton(

            self.sidebar,

            text="Dashboard",

            command=lambda:
                self.show_page(
                    "dashboard"
                )
        )

        self.btn_dashboard.pack(
            fill="x",
            padx=10,
            pady=5
        )

        self.btn_dataset = ctk.CTkButton(

            self.sidebar,

            text="Datasets",

            command=lambda:
                self.show_page(
                    "datasets"
                )
        )

        self.btn_dataset.pack(
            fill="x",
            padx=10,
            pady=5
        )

        self.btn_training = ctk.CTkButton(

            self.sidebar,

            text="Training",

            command=lambda:
                self.show_page(
                    "training"
                )
        )

        self.btn_training.pack(
            fill="x",
            padx=10,
            pady=5
        )

        self.btn_results = ctk.CTkButton(

            self.sidebar,

            text="Results",

            command=lambda:
                self.show_page(
                    "results"
                )
        )

        self.btn_results.pack(
            fill="x",
            padx=10,
            pady=5
        )

        self.btn_deployment = ctk.CTkButton(

            self.sidebar,

            text="Deployment",

            command=lambda:
                self.show_page(
                    "deployment"
                )
        )

        self.btn_deployment.pack(
            fill="x",
            padx=10,
            pady=5
        )

        self.btn_settings = ctk.CTkButton(

            self.sidebar,

            text="Settings",

            command=lambda:
                self.show_page(
                    "settings"
                )
        )

        self.btn_settings.pack(
            fill="x",
            padx=10,
            pady=5
        )

        # ----------------------------------------------------
        # Main Content Area
        # ----------------------------------------------------

        self.content = ctk.CTkFrame(
            self
        )

        self.content.grid(

            row=0,

            column=1,

            sticky="nswe"
        )

        self.content.grid_rowconfigure(
            0,
            weight=1
        )

        self.content.grid_columnconfigure(
            0,
            weight=1
        )

        self.pages = {

            "dashboard":
                DashboardPage(
                    self.content
                ),

            "datasets":
                DatasetPage(
                    self.content
                ),

            "training":
                TrainingPage(
                    self.content
                ),

            "results":
                ResultsPage(
                    self.content
                ),

            "deployment":
                DeploymentPage(
                    self.content
                ),

            "settings":
                SettingsPage(
                    self.content
                )
        }

        self.current_page = None

        self.show_page(
            "dashboard"
        )

    # ========================================================
    # Page Switcher
    # ========================================================

    def show_page(
        self,
        page_name
    ):

        if self.current_page:

            self.current_page.grid_forget()

        self.current_page = self.pages[
            page_name
        ]

        self.current_page.grid(

            row=0,

            column=0,

            sticky="nswe"
        )

# ============================================================
# Launch
# ============================================================

def run():

    setup_theme()

    app = MESHApp()

    app.mainloop()

if __name__ == "__main__":

    run()