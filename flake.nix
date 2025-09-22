{
  description = "Python development environment with essential packages from Nix and uv for flexibility";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, nixpkgs-unstable }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pkgs-unstable = nixpkgs-unstable.legacyPackages.${system};
        
        # Define essential Python packages that should come from Nix
        python-with-packages = pkgs.python312.withPackages (ps: with ps; [
          ipython
          python-lsp-ruff
          jupyter
          python-lsp-server
          jupyter-lsp
          numpy
          playwright
          # Add more essential packages here as needed
        ]);
      in
      {
        devShells.default = (pkgs.buildFHSEnv {
          name = "uv-env";

          targetPkgs = pkgs: with pkgs; [
            # Python with essential packages
            python-with-packages
            
            # uv for additional package management
            uv
            stdenv.cc.cc.lib  # Provides libstdc++.so.6 and other C++ stdlib
            zlib
            ruff
            pyright
            nodejs_24
            pkgs-unstable.opencode
            playwright
            ffmpeg
          ];

          profile = ''
            export PYTHONPATH="${python-with-packages}/${python-with-packages.sitePackages}"
            
            # Set up some helpful aliases
            alias python=python3
          '';

          runScript = ''

            echo "Setting up Python development environment..."
            echo "Python with essential packages: $(python --version)"
            
            # Create .venv if it doesn't exist
            if [ ! -d ".venv" ]; then
              echo "Creating uv virtual environment..."
              uv venv --python $(which python) --system-site-packages
            fi
            
            # Activate the virtual environment
            echo "Activating virtual environment..."
            source .venv/bin/activate

            # Ensure uv uses the virtual environment
            export UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV"
            
            # Install additional dependencies if pyproject.toml exists
            if [ -f "pyproject.toml" ]; then
              echo "Installing additional dependencies with uv..."
              uv sync
            elif [ -f "requirements.txt" ]; then
              echo "Installing additional dependencies from requirements.txt..."
              uv pip install -r requirements.txt
            fi

            python -m ipykernel install --user --name=uv-env --display-name "Python (uv)"
            
            echo ""
            echo "Development environment ready!"
            echo "Python interpreter: $(which python)"
            echo "Virtual environment: $VIRTUAL_ENV"

            fish
          '';

        }).env;
      });
}
