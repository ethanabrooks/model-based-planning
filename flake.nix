{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/a983cc62cc2345443620597697015c1c9c4e5b06";
    utils.url = "github:numtide/flake-utils/93a2b84fc4b70d9e089d029deacc3583435c2ed6";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
  }: let
    out = system: let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
      };
      inherit (pkgs) poetry2nix;
      mujoco = fetchTarball {
        url = "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz";
        sha256 = "sha256:1lvppcdfca460sqnb0ryrach6lv1g9dwcjfim0dl4vmxg2ryaq7p";
      };
      overrides = pyfinal: pyprev: rec {
        mjrl = pyprev.buildPythonPackage {
          pname = "mjrl";
          version = "0.0.0";
          buildInputs = with pyfinal; [
            setuptools
            gym
            six
            mujoco-py
          ];
          src = fetchGit {
            url = "https://github.com/aravindr93/mjrl";
            rev = "3871d93763d3b49c4741e6daeaebbc605fe140dc";
          };
          doCheck = false;
        };
        d4rl =
          (pyprev.d4rl.override {
            preferWheel = false;
          })
          .overridePythonAttrs (old: {
            buildInputs = old.buildInputs ++ [pyfinal.mjrl];
            patches = [./d4rl.patch];
          });
        mujoco-py =
          (pyprev.mujoco-py.override {
            preferWheel = false;
          })
          .overridePythonAttrs (old: {
            env.NIX_CFLAGS_COMPILE = "-L${pkgs.mesa.osmesa}/lib";
            preBuild = ''
              export MUJOCO_PY_MUJOCO_PATH="${mujoco}"
              export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${mujoco}/bin:${pkgs.mesa.osmesa}/lib:${pkgs.libGL}/lib:${pkgs.gcc-unwrapped.lib}/lib
            '';
            buildInputs =
              old.buildInputs
              ++ [
                pyfinal.setuptools
                pkgs.mesa
                pkgs.libGL
              ];
            patches = [./mujoco-py.patch];
          });
        torch = pyprev.pytorch-bin.overridePythonAttrs (old: {
          src = pkgs.fetchurl {
            url = "https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp39-cp39-linux_x86_64.whl";
            sha256 = "sha256-20V6gi1zYBO2/+UJBTABvJGL3Xj+aJZ7YF9TmEqa+sU=";
          };
        });
        torchrl = pyprev.torchrl.overridePythonAttrs (old: {
          preFixup = "addAutoPatchelfSearchPath ${pyfinal.torch}";
        });
      };
      poetryEnv = pkgs.poetry2nix.mkPoetryEnv {
        python = pkgs.python39;
        projectDir = ./.;
        preferWheels = true;
        overrides = poetry2nix.overrides.withDefaults overrides;
      };
    in {
      devShell = pkgs.mkShell {
        LD_LIBRARY_PATH = "$LD_LIBRARY_PATH:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.mesa.osmesa}/lib:${pkgs.libGL}/lib:${pkgs.gcc-unwrapped.lib}/lib";
        buildInputs = with pkgs; [
          alejandra
          poetry
          poetryEnv
        ];
        PYTHONBREAKPOINT = "ipdb.set_trace";
        shellHook = ''
          set -o allexport
          source .env
          set +o allexport
        '';
      };
    };
  in
    utils.lib.eachDefaultSystem out;
}
