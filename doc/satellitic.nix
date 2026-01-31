{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python313;

  pyPkgs = python.pkgs.overrideScope (self: super: {

    satellitic = super.buildPythonPackage rec {
      pname = "satellitic";
      version = "0.1.7";
      format = "pyproject";

      src = super.fetchPypi {
        inherit pname version;
        sha256 = "kHFWVnlA9gnJabqpORtNZCHO5vWm8qIHSywIM3A+LBo=";
      };

      nativeBuildInputs = with super; [
        hatchling
        numpy
      ];

      propagatedBuildInputs = with super; [
        numpy
        pandas
        jax
        sgp4
        vispy
        astropy
        matplotlib
        requests
      ];

      pythonImportsCheck = [ "satellitic" ];
    };

    impetuous = super.buildPythonPackage rec {
      pname = "impetuous-gfa";
      version = "0.101.15";
      format = "pyproject";

      src = super.fetchPypi {
        inherit pname version;
        sha256 = "tE1ktjJzAZXw0LfGMz5IR9nFrXyDnbMZh0emA0N7L74=";
      };

      #nativeBuildInputs = with super; [
      #  hatchling
      #  numpy
      #];
      nativeBuildInputs = with super; [
        setuptools
        wheel
      ];

      propagatedBuildInputs = with super; [
        numpy
        pandas
        scipy
        scikit-learn
        statsmodels
      ];
    };

    miepython = super.buildPythonPackage rec {
      pname = "miepython";
      version = "3.0.2";
      format = "pyproject";

      src = super.fetchPypi {
        inherit pname version;
        sha256 = "II7Ool1CJBZQ5pRxc4cWlfxn5EkyYFReUZnKhf/XvJo=";
      };

      #nativeBuildInputs = with super; [
      #  hatchling
      #  numpy
      #];
      nativeBuildInputs = with super; [
        setuptools
        wheel
      ];

      propagatedBuildInputs = with super; [
        numpy
        scipy
        numba
        matplotlib
      ];
    };
  });

  pythonEnv = python.withPackages (ps: with ps; [
    pip
    ipython

    numpy
    pandas
    scipy
    numba
    scikit-learn
    statsmodels

    jax
    sgp4
    vispy
    astropy

    pyPkgs.satellitic
    pyPkgs.impetuous
    pyPkgs.miepython
  ]);

in pkgs.mkShell {
  name = "satellitic-dev-shell";

  buildInputs = [
    pythonEnv
  ];

  shellHook = ''
    export PYTHONNOUSERSITE=1
    export SOURCE_DATE_EPOCH=$(date +%s)

    echo "🚀 satellitic dev shell"
    echo "Python: $(python --version)"
    echo "satellitic import test:"
    python - <<EOF
import satellitic
print("OK:", satellitic.__version__)
EOF
  '';
}
