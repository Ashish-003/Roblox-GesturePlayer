#!/usr/bin/python3

#
# This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# https://creativecommons.org/licenses/by-nc-nd/4.0/
#

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Union, List

WINE_VERSIONS = {
    "6.16.r3.gf3b03ce5": {
        "url": "https://cdn.discordapp.com/attachments/858117357897121822/884173115398176768/debuntu-wine-tkg-staging-fsync-git-6.16.r3.gf3b03ce5.7z",
        "hash": {
            "type": "sha512",
            "value": "e0e1bc6ecc0226f2a2f8e1ddbff28d36b5836c2ff2e50f733fbb86fcaacb50097f6f47281801c91b8412801a993a354b4aca24978eb2d86e7f64ee2fcf543b43"
        }
    },
    "6.17.r0.g5f19a815": {
        "url": "https://cdn.discordapp.com/attachments/858117357897121822/886180600703250453/debuntu-wine-tkg-staging-fsync-git-6.17.r0.g5f19a815.7z",
        "hash": {
            "type": "sha512",
            "value": "bd64593b2f3a01942bc5c6c5ee6aa654f5cce30c76e4a83397611286cd74feaf7ba8fb6fba99f145bb4d9eed523a1d007cc4d845d5324b07a40700dcf1655a3b"
        }
    },
    "6.18.r0.gf8851f16": {
        "url": "https://cdn.discordapp.com/attachments/858117357897121822/891630589671796786/debuntu-wine-tkg-staging-fsync-git-6.18.r0.gf8851f16.7z",
        "hash": {
            "type": "sha512",
            "value": "223f81f559e84f1b1915bdb71470dfed2648ddf60650e9c1ee6680f325cd7ebf73b1b98e0ab3d3348a069bba36775acffd157403aa453a3bf14f268500ed0dbe"
        }
    },
    "6.20.r0.g3fb6eb99": {
        "url": "https://cdn.discordapp.com/attachments/858117357897121822/901364156597428274/debuntu-wine-tkg-staging-fsync-git-6.20.r0.g3fb6eb99.7z",
        "hash": {
            "type": "sha512",
            "value": "8a20ccf11ae4e63c9bf382437e8dad7bc02041884784624b84152b93beb849463276e4503a674339586a4a75a541e6c49b2aab0110a13c80ce36249bc124e73e"
        }
    },
    "6.22.r0.ga703038b": {
        "url": "https://cdn.discordapp.com/attachments/858117357897121822/912141274805575710/debuntu-wine-tkg-staging-fsync-git-6.22.r0.ga703038b.7z",
        "hash": {
            "type": "sha512",
            "value": "88112abe39c03a76dbfb8c64e0f48d02576c6920114e46141d096f947e5f1c774051022fd6d0101e333cd30de810b4d3435d66184fb752da4870954e8c419cac"
        }
    },
    "7.0rc1.r0.g544f90da": {
        "url": "https://cdn.discordapp.com/attachments/858117357897121822/920772054784016444/debuntu-wine-tkg-staging-fsync-git-7.0rc1.r0.g544f90da.7z",
        "hash": {
            "type": "sha512",
            "value": "ca1da05759622762e1e54bae1d067387aa98fa31073f6b52c51f76a6f5c23f1b7d03fdb6dfb35d2c29609a5fc19e0a64c6f034fc67b47d6f4094ad529d97641a"
        }
    },
    "7.0rc2.r0.g8f579c4e": {
        "url": "https://cdn.discordapp.com/attachments/858117357897121822/922274716096528494/debuntu-wine-tkg-staging-fsync-git-7.0rc2.r0.g8f579c4e.7z",
        "hash": {
            "type": "sha512",
            "value": "66d32f90af8f5de11ace6b930dd5a989a8f0a5d9e09244da81adfad2bb68bbe596e8259ecec49682c34fcaa03ce2e98501b05c35a8a1da7d9881c97599eb7fbe"
        }
    },
    "7.0rc4.r1.g98c906f8": {
        "url": "https://cdn.discordapp.com/attachments/858117357897121822/928092680603435029/debuntu-wine-tkg-staging-fsync-git-7.0rc4.r1.g98c906f8.7z",
        "hash": {
            "type": "sha512",
            "value": "2451843024f195295a73e656cb3e683de4a1eb94765a70d9c59be3de3580652dc11bb0bc8080b7f63238c0c14c763f2af62b6a270726fa6441e35e84634dee7b"
        }
    },
    "7.0rc6.r0.g0111d074": {
        "url": "https://cdn.discordapp.com/attachments/858117357897121822/932136639168380928/debuntu-wine-tkg-staging-fsync-git-7.0rc6.r0.g0111d074.7z",
        "hash": {
            "type": "sha512",
            "value": "3e30e65abeee8f15b5c7547fb4232b611be27aa6505746cd25b4cc24b389b097443a5446b4d52c2d8ea4dd1415c09262f3595a1f03375d0102c4e47a81a7b5ea"
        }
    },
    "7.0.r1.g95bf6698": {
        "url": "https://cdn.discordapp.com/attachments/858117357897121822/934769842832146494/debuntu-wine-tkg-staging-fsync-git-7.0.r1.g95bf6698.7z",
        "hash": {
            "type": "sha512",
            "value": "f28fe737a358b5a9b6d89e3baef85e6168a23598eb327c378a2a284bc2fd959331e8637a64db59be75db35d79ab815e2688055d3de5544c9b653e5907389a2b6"
        }
    },
    "7.1.r2.gc437a01e": {
        "url": "https://cdn.discordapp.com/attachments/858117357897121822/937332523380379678/debuntu-wine-tkg-staging-fsync-git-7.1.r2.gc437a01e.7z",
        "hash": {
            "type": "sha512",
            "value": "9b0da4604525b88f2ed353780211a4d78c464326cffdd2d7dfdc5d487a3c9631de9f0197c2f63bd9c1475e5e48ff7d9783e9648582aa8dc29b58d6e62f617c49"
        }
    },
    "7.2.r0.g68441b1d": {
        "url": "https://cdn.discordapp.com/attachments/858117357897121822/943053350528778240/debuntu-wine-tkg-staging-fsync-git-7.2.r0.g68441b1d.7z",
        "hash": {
            "type": "sha512",
            "value": "78a5227807eb1af4656f4e31f2989c87eabd4754319f44d8b4b9db02a34b75569c2a0b401e5378eb215b56324f8cd8a7f7661f9e5a82bb33d667542f5b838ec6"
        }
    },
    "7.6.r14.g6b4b9f1b": {
        "url": "https://cdn.discordapp.com/attachments/858117357897121822/967108819849199667/debuntu-wine-tkg-staging-fsync-git-7.6.r14.g6b4b9f1b.7z",
        "hash": {
            "type": "sha512",
            "value": "eaa52d360e92eb7715b0db8b414ba0ba15b4cba8e168355ada5d171d6c0f47b4ef8f85fe433e39a152d239b9d03fbe4a9c11851c86415d65c4cd2658560faf49"
        }
    },
    "7.7.r0.g9df73ee3": {
        "url": "https://cdn.discordapp.com/attachments/858117357897121822/967687344796893285/debuntu-wine-tkg-staging-fsync-git-7.7.r0.g9df73ee3.7z",
        "hash": {
            "type": "sha512",
            "value": "019ce0d97f7b6f504db51954bc79761d078062c1a7b69f349de380b2a9c9081de11b6f1cabe9665cbcee28ebb9a20cd57519bc0dc795401d80a6b55a04efe3ed"
        }
    }
}


def use_version():
    return "7.7.r0.g9df73ee3"


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def info(msg: str):
    print(f"{bcolors.BOLD}>> {bcolors.OKBLUE}{msg}{bcolors.ENDC}{bcolors.ENDC}")


def warn(msg: str):
    print(f"{bcolors.BOLD}>> {bcolors.WARNING}{msg}{bcolors.ENDC}{bcolors.ENDC}")


def success_message(msg: str):
    print(f"{bcolors.BOLD}>>> {bcolors.OKGREEN}{msg}{bcolors.ENDC}{bcolors.ENDC}")


def error_out(msg: str):
    print(
        f"{bcolors.BOLD}>>> {bcolors.FAIL}{msg}{bcolors.ENDC}{bcolors.ENDC}",
        file=sys.stderr
    )
    print(
        f"{bcolors.BOLD}>>> {bcolors.FAIL}Quitting script due to an error.{bcolors.ENDC}{bcolors.ENDC}",
        file=sys.stderr
    )
    sys.exit(-1)


def info_on_call(message: str):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            args_as_string = json.dumps(list(map(str, args)))
            kwargs_as_string = json.dumps(dict(zip(map(str, kwargs.keys()), map(str, kwargs.values()))))

            info(f"* {message} :: {bcolors.WARNING}{args_as_string} :: {kwargs_as_string}{bcolors.ENDC}")
            return_value = fn(*args, **kwargs)

            info(f"* {bcolors.OKGREEN}{str(return_value)}{bcolors.ENDC}")

            return return_value

        return wrapper

    return decorator


def download_data():
    return WINE_VERSIONS.get(use_version())


def download_url():
    return download_data()["url"]


def filename():
    return download_url().split("/")[-1]


def download_hash_algorithm():
    return download_data()["hash"]["type"]


def download_hash_value():
    return download_data()["hash"]["value"]


def download_path() -> Path:
    return Path("/", "tmp", filename())


def share_grapejuice_path() -> Path:
    return Path(os.environ["HOME"]).resolve() / ".local" / "share" / "grapejuice"


def grapejuice_user_path() -> Path:
    return share_grapejuice_path() / "user"


def wine_target_path() -> Path:
    p = grapejuice_user_path() / "wine-download"
    p.mkdir(parents=True, exist_ok=True)

    return p


def find_latest_previous_wine_server() -> Union[Path, None]:
    candidates = list(
        sorted(
            filter(
                lambda p: f"bin{os.sep}wineserver" in str(p),
                wine_target_path().rglob("wineserver")
            )
        )
    )

    if len(candidates) == 0:
        return None

    return candidates[-1]


def grapejuice_settings_path() -> Path:
    return Path(os.environ["HOME"]).resolve() / ".config" / "brinkervii" / "grapejuice" / "user_settings.json"


@info_on_call("Hashing file")
def hash_file(path: Path, algorithm: str, block_size: int = 4096) -> str:
    h = hashlib.new(algorithm)

    with path.open("rb") as fp:
        data = True

        while data:
            data = fp.read(block_size)
            h.update(data)

    return h.hexdigest().lower().strip()


@info_on_call("Locating system binary")
def which(binary_name: str, path_extra: List[Union[Path, str]] = None) -> Union[Path, None]:
    path = list(map(str, [] if path_extra is None else path_extra))
    path.extend(os.environ.get("PATH", "").split(":"))

    for d in path:
        d = d.strip()
        if not d:
            continue

        d = Path(d).resolve()
        file = d / binary_name

        if file.exists() and (file.is_file() or file.is_symlink()):
            return file

    return None


def stop_wine_server():
    info("Stopping Wine server")

    prefixes_directory = share_grapejuice_path() / "prefixes"

    if prefixes_directory.exists():
        info("Using Grapejuice 4+ prefix configuration")
        prefix_list = list(filter(Path.is_dir, prefixes_directory.glob("*")))

    else:
        info("Using Grapejuice <= 3 prefix directory")
        prefix_list = [share_grapejuice_path() / "wineprefix"]

    for prefix_path in prefix_list:
        wine_server_path = find_latest_previous_wine_server() or which("wineserver")
        if wine_server_path is None:
            info("Could not find wineserver, so it is not being stopped")
            return

        if prefix_path.exists() and prefix_path.is_dir():
            try:
                subprocess.call([str(wine_server_path), "-k"], env={"WINEPREFIX": str(prefix_path)})

            except Exception as e:
                warn(f"Failed to stop wineserver: {str(e)}")
                pass


def wget_bin():
    p = which("wget")
    if p is None:
        error_out("The 'wget' binary is not present on your system, please install wget.")

    return p


@info_on_call("Downloading file using wget")
def wget(url: str, download_location: Path, hash_algorithm: str, file_hash: str):
    if download_location.exists():
        if download_location.is_dir():
            error_out(f"The wine download location {download_location} is a directory and cannot be written to. "
                      f"Please move or delete the directory.")

        else:
            hash_value = hash_file(download_location, hash_algorithm)
            if hash_value == file_hash:
                return

            else:
                warn(f"File exists at {download_location}, but the file hash does not match. Attempting redownload.")

    subprocess.check_call([str(wget_bin()), url, "-O", str(download_location)])

    hash_value = hash_file(download_location, hash_algorithm)

    if hash_value != file_hash:
        error_out(f"The Wine build was downloaded to {download_location}. However, the file hash does not match. "
                  f"Please make sure the download has finished and that your internet connection is secure!")


def seven_zip_bin() -> Path:
    p = which("7z") or which("7za")

    if p is None:
        error_out("The '7z' binary is not present on your system, please install 7z (the package is called 'p7zip' or "
                  "'p7zip-full' on most distributions).")

    return p


@info_on_call("Locating file in 7z archive")
def find_wine_binary_in_7z(source: Path) -> str:
    listing = subprocess.check_output([str(seven_zip_bin()), "l", str(source)]).decode("UTF-8")

    candidates = list(filter(lambda s: s.strip("\r").strip().endswith("bin/wine"), listing.split("\n")))
    assert len(candidates) == 1, "Invalid archive, invalid number of wine binary candidates"

    m = re.split(r"\s+", candidates[0])
    p = m[-1]

    assert p.endswith("bin/wine"), "Got an invalid wine binary path from archive"

    return p


@info_on_call("Unarchiving 7z archive")
def unarchive_7z(source: Path, target: Path):
    subprocess.check_call([str(seven_zip_bin()), "x", "-y", "-o" + str(target), str(source)])


@info_on_call("Updating Grapejuice settings")
def update_grapejuice_settings(wine_binary_path: str):
    if not grapejuice_settings_path().exists():
        error_out("The Grapejuice settings file does not exist. Please open and close Grapejuice one time.")

    with grapejuice_settings_path().open("r") as fp:
        data = json.load(fp)

    full_wine_binary_path = wine_target_path() / wine_binary_path
    settings_version = data.get("__version__", 0)

    if settings_version >= 2:
        info("Got Grapejuice 4+ user_settings")
        wine_home = full_wine_binary_path.parent.parent
        wine_bin = wine_home / "bin"

        if not wine_bin.exists():
            error_out(f"Invalid wine_home: {wine_home}")

        for prefix in data.get("wineprefixes", []):
            prefix["wine_home"] = str(wine_home)

        data["default_wine_home"] = str(wine_home)

    else:
        info("Got Grapejuice < 4 user_settings")
        data["wine_binary"] = str(full_wine_binary_path)

    with grapejuice_settings_path().open("w") as fp:
        json.dump(data, fp, indent=2)


def main():
    info("Starting installation process")
    stop_wine_server()

    download_location = download_path()
    wget(download_url(), download_location, download_hash_algorithm(), download_hash_value())

    info("Processing archive")
    wine_binary_path = find_wine_binary_in_7z(download_location)
    unarchive_7z(download_location, wine_target_path())

    update_grapejuice_settings(wine_binary_path)

    stop_wine_server()
    info("Done")

    success_message("Wine-TKG installation has succeeded!")


if __name__ == '__main__':
    main()
