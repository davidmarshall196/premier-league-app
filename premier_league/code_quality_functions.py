from typing import Union, List
import subprocess
import pytest


def run_unit_tests(
    test_file: str,
    src_folder: str,
    cov_folder: str = None,
    exclude_files: List[str] = [],
) -> Union[int, None]:
    """
    Run unit tests using pytest.

    Parameters:
    - test_file (str): The path to the test file to run.
    - src_folder (str): The path to the source code folder.
    - cov_folder (str, optional): The folder to generate coverage for. Defaults to None.
    - exclude_files (List[str], optional): List of file paths to exclude.
    Defaults to an empty list.

    Returns:
    - int or None: Returns the return code of pytest, or None if an error occurred.
    """
    try:
        pytest_command = [
            test_file,
            "-v",
            "-p",
            "no:cacheprovider",
            "--disable-pytest-warnings",
        ]

        if cov_folder:
            pytest_command.append(f"--cov={cov_folder}")
            if exclude_files:
                # Join the list of files to exclude into a string separated by commas
                excluded_files_str = ",".join(exclude_files)
                pytest_command.append(f"--cov-omit={excluded_files_str}")
            pytest_command.append("--cov-report=term-missing")

        # Run pytest
        retcode = pytest.main(pytest_command)

        assert retcode == 0, "Tests failed. Please check."

        return retcode

    except Exception as e:
        print(f"An error occurred while running tests: {e}")
        return None


def run_code_quality_checks(
    src_folder: str, exclude_folder: str = None
) -> Union[str, None]:
    """
    Run code quality checks using the Black formatter and Flake8 linter.

    Parameters:
    - src_folder (str): The path to the source code folder to check.
    - exclude_folder (str, optional): The path to the folder to exclude.

    Returns:
    - str or None: Returns error messages if any, otherwise None.
    """
    errors = []
    try:
        black_cmd = ["black", src_folder]
        if exclude_folder:
            black_cmd.extend(["--exclude", exclude_folder])
        black = subprocess.run(black_cmd, capture_output=True, text=True, check=True)
        print("Black Output:")
        print(black.stderr)

        if black.stderr:
            errors.append(black.stderr)

        flake8_cmd = [
            "flake8",
            "--extend-ignore=E203",
            "--max-line-length=88",
            src_folder,
        ]
        if exclude_folder:
            flake8_cmd.extend([f"--exclude={exclude_folder}"])
        flakey = subprocess.run(flake8_cmd, capture_output=True, text=True)
        print("\nFlake8 Output:")
        if flakey.stdout == "":
            print("Done. Flake8 has no improvements!")
        else:
            print(flakey.stdout)

        if flakey.stdout:
            errors.append(flakey.stdout)

        if errors:
            return "\n".join(errors)

        return None

    except subprocess.CalledProcessError as e:
        error_message = f"An error occurred while running {e.cmd[0]}: {e.stderr}"
        return error_message
