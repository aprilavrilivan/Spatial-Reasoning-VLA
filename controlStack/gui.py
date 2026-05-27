import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from unit_test_runner import UNIT_TESTS, UnitTestRunner
from vlm_client import DEFAULT_ADAPTER_PATH


class SpatialVLAGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Spatial-VLA Robot Unit Tests")
        self.geometry("520x360")
        self.resizable(False, False)
        self.runner = None

        self.test_name = tk.StringVar(value=sorted(UNIT_TESTS)[0])
        self.trials = tk.IntVar(value=10)
        self.camera_id = tk.IntVar(value=0)
        self.adapter_path = tk.StringVar(value=DEFAULT_ADAPTER_PATH)
        self.use_adapter = tk.BooleanVar(value=True)
        self.execute_actions = tk.BooleanVar(value=False)
        self.show_camera = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Ready.")

        self._build()

    def _build(self):
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Unit test").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Combobox(
            frame,
            textvariable=self.test_name,
            values=sorted(UNIT_TESTS),
            state="readonly",
            width=36,
        ).grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(frame, text="Trials").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Spinbox(frame, from_=1, to=100, textvariable=self.trials, width=8).grid(row=1, column=1, sticky="w", pady=4)

        ttk.Label(frame, text="Camera ID").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Spinbox(frame, from_=0, to=8, textvariable=self.camera_id, width=8).grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(frame, text="Adapter path").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.adapter_path, width=40).grid(row=3, column=1, sticky="ew", pady=4)

        ttk.Checkbutton(
            frame,
            text="Use fine-tuned adapter",
            variable=self.use_adapter,
        ).grid(row=4, column=1, sticky="w", pady=4)
        ttk.Checkbutton(
            frame,
            text="Send movement commands to robot",
            variable=self.execute_actions,
        ).grid(row=5, column=1, sticky="w", pady=8)
        ttk.Checkbutton(
            frame,
            text="Show camera frame",
            variable=self.show_camera,
        ).grid(row=6, column=1, sticky="w", pady=4)

        ttk.Button(frame, text="Run Test", command=self.run_async).grid(row=7, column=1, sticky="e", pady=16)
        ttk.Label(frame, textvariable=self.status, wraplength=460).grid(row=8, column=0, columnspan=2, sticky="w", pady=8)

        frame.columnconfigure(1, weight=1)

    def run_async(self):
        if self.runner is not None:
            messagebox.showinfo("Spatial-VLA", "A test is already running.")
            return

        thread = threading.Thread(target=self._run_test, daemon=True)
        thread.start()

    def _run_test(self):
        try:
            self.status.set("Initializing camera/model...")
            output_dir = Path("real_robot_results") / "unit_tests"
            self.runner = UnitTestRunner(
                camera_id=self.camera_id.get(),
                adapter_path=self.adapter_path.get(),
                use_adapter=self.use_adapter.get(),
                run_label="finetuned_qwen" if self.use_adapter.get() else "base_qwen",
                output_dir=output_dir,
                execute_actions=self.execute_actions.get(),
                show_camera=self.show_camera.get(),
            )
            self.status.set(f"Running {self.test_name.get()} for {self.trials.get()} trials...")
            self.runner.run(self.test_name.get(), self.trials.get())
            self.status.set("Finished. Results saved under real_robot_results/unit_tests.")
        except Exception as exc:
            self.status.set(f"Error: {exc}")
            messagebox.showerror("Spatial-VLA", str(exc))
        finally:
            self.runner = None


if __name__ == "__main__":
    SpatialVLAGui().mainloop()
