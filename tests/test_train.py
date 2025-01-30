import subprocess
import re

def test_training_script():
    """
    Vérifie que train.py s'exécute et produit une accuracy >= 0.8
    """
    cmd = ["pipenv", "run", "python", "src/train.py"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Vérification de la terminaison sans erreur
    assert result.returncode == 0, f"train.py failed:\n{result.stderr}"

    # Rechercher "Accuracy = 0.9..." dans la sortie
    match = re.search(r"Accuracy = (\d\.\d+)", result.stdout)
    assert match is not None, "No accuracy found in output"
    acc = float(match.group(1))
    assert acc >= 0.8, f"Accuracy is below 0.8: {acc}"
