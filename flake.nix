{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/";
    utils.url = "github:numtide/flake-utils/";
    nixgl.url = "github:guibou/nixGL";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
    nixgl,
  }: let
    out = system: let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
        overlays = [nixgl.overlay];
      };
      inherit (pkgs) poetry2nix;
      mujoco = pkgs.stdenv.mkDerivation rec {
        pname = "mujoco";
        version = "2.1.0";

        src = fetchTarball {
          url = "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz";
          sha256 = "sha256:1lvppcdfca460sqnb0ryrach6lv1g9dwcjfim0dl4vmxg2ryaq7p";
        };
        installPhase = ''
          install -d $out/bin
          install -m 755 bin/* $out/bin/
          install -d $out/include
          install include/* $out/include/
        '';
        nativeBuildInputs = with pkgs;
          [glew110 autoPatchelfHook]
          ++ (with pkgs.xorg; [
            libXcursor
            libXinerama
            libXrandr
            libXxf86vm
          ]);
      };

      python = pkgs.python39;
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
            preBuild = with pkgs; ''
              export MUJOCO_PY_MUJOCO_PATH="${mujoco}"
              export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${mujoco}/bin:${mesa.osmesa}/lib:${libGL}/lib:${gcc-unwrapped.lib}/lib:${glew110}/lib
            '';
            buildInputs =
              old.buildInputs
              ++ [pyfinal.setuptools]
              ++ (with pkgs; [mesa libGL]);
            patches = [./mujoco-py.patch];
          });
        # Based on https://github.com/NixOS/nixpkgs/blob/nixos-22.11/pkgs/development/python-modules/torch/bin.nix#L107
        torch = pyprev.buildPythonPackage {
          version = "1.13.1";

          pname = "torch";
          # Don't forget to update torch to the same version.

          format = "wheel";

          src = pkgs.fetchurl {
            url = "https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp39-cp39-linux_x86_64.whl";
            sha256 = "sha256-20V6gi1zYBO2/+UJBTABvJGL3Xj+aJZ7YF9TmEqa+sU=";
          };

          # extract wheel, run normal patch phase, repack wheel.
          # effectively a "wheelPatchPhase". not a normal thing
          # to do but needs must.
          patchPhase = ''
            wheelFile="$(realpath -s dist/*.whl)"
            pushd "$(mktemp -d)"

            unzip -q "$wheelFile"

            patchPhase

            newZip="$(mktemp -d)"/new.zip
            zip -rq "$newZip" *
            rm -rf "$wheelFile"
            cp "$newZip" "$wheelFile"

            popd
          '';

          nativeBuildInputs = with pkgs; [
            addOpenGLRunpath
            patchelf
            unzip
            zip
          ];

          propagatedBuildInputs = with pyfinal; [
            future
            numpy
            pyyaml
            requests
            setuptools
            typing-extensions
          ];

          postInstall = ''
            # ONNX conversion
            rm -rf $out/bin
          '';

          postFixup = let
            rpath = with pkgs; lib.makeLibraryPath [stdenv.cc.cc.lib];
          in ''
            find $out/${python.sitePackages}/torch/lib -type f \( -name '*.so' -or -name '*.so.*' \) | while read lib; do
              echo "setting rpath for $lib..."
              patchelf --set-rpath "${rpath}:$out/${python.sitePackages}/torch/lib" "$lib"
              addOpenGLRunpath "$lib"
            done
          '';

          # The wheel-binary is not stripped to avoid the error of `ImportError: libtorch_cuda_cpp.so: ELF load command address/offset not properly aligned.`.
          dontStrip = true;

          pythonImportsCheck = ["torch"];
        };

        torchrl = pyprev.buildPythonPackage {
          pname = "torchrl";
          version = "0.0.5";
          propagatedBuildInputs = with pyfinal; [
            tensordict
          ];
          nativeBuildInputs = with pyfinal; [
            setuptools
            cloudpickle
            packaging
            pkgs.which
          ];
          src = pkgs.fetchgit {
            url = "https://github.com/pytorch/rl";
            rev = "48eca9890e24a64b0f3ee133a74abafad7cd0768";
            sha256 = "sha256-ZLHqKRO7O4ZOSPqtwVmyFElpRnEbhX97Iy8PkGh4Fqg=";
            name = "torchrl";
          };
          doCheck = false;
          patches = [./torchrl.patch];
        };
      };
      poetryEnv = pkgs.poetry2nix.mkPoetryEnv {
        inherit python;
        projectDir = ./.;
        preferWheels = true;
        overrides = poetry2nix.overrides.withDefaults overrides;
      };

      myNixgl = pkgs.nixgl.override {
        nvidiaVersion = "510.108.03";
        nvidiaHash = "sha256-QQpRXnjfKcLLpKwLSXiJzg/xsEz8cR/4ieLfyA8NoNg=";
      };
    in {
      devShell = pkgs.mkShell {
        MUJOCO_PY_MUJOCO_PATH = "${mujoco}";
        LD_LIBRARY_PATH = with pkgs; "$LD_LIBRARY_PATH:${mesa.osmesa}/lib:${gcc-unwrapped.lib}/lib:${mujoco}/bin";
        buildInputs = with pkgs; [
          alejandra
          poetry
          poetryEnv
          ffmpeg
          myNixgl.nixGLNvidia
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
