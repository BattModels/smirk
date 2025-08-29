import inspect
import logging
from pathlib import Path


def linkcode_resolve(domain: str, info: dict):
    if domain == "py":
        try:
            return resolve_py(info)
        except Exception as e:
            logging.error(e)

    return None


linkcode_url = (
    "https://github.com/BattModels/smirk/blob/main/{filepath}#L{linestart}-L{linestop}"
)

GIT_ROOT = Path(__file__).parent.parent


def resolve_py(info):
    module_name = info.get("module")
    fullname = info.get("fullname")
    if not module_name or not fullname:
        return None
    try:
        obj = __import__(module_name)
        for part in fullname.split("."):
            obj = getattr(obj, part)
        source_file = inspect.getsourcefile(obj)
        source_lines, lineno = inspect.getsourcelines(obj)
        linestop = lineno + len(source_lines) - 1
    except Exception as e:
        logging.error(e)
        return None

    if not source_file or not lineno:
        return None
    source_file = Path(source_file).relative_to(GIT_ROOT)
    return linkcode_url.format(
        filepath=source_file, linestart=lineno, linestop=linestop
    )


if __name__ == "__main__":
    print(
        resolve_py(
            {
                "module": "smirk",
                "fullname": "SmirkTokenizerFast.is_fast",
            }
        )
    )
