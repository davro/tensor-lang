# ================================================================
#                          TensorLang
# ================================================================

class TensorLang:
    def __init__(self, width: int = 80):
        self.width = width

    def print_header(self, version: str):
        border = "-" * (self.width - 2)
        title = f"TensorLang Compiler v{version}"

        print(f"+{border}+")
        print(f"+{border}+")
        print(f"{'|                          '}{title}                          |")
        print(f"+{border}+")
        print(f"+{border}+")
        #print("")

    def print(self, type: str="" , message: str=""):
        if not type:
            type = "[DEBUG]"
        print(f"{type} {message}")

    def seperator(self):
        print("-" * self.width)
