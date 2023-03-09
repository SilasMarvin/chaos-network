{
  description = "ad";

  inputs = {
    nixpkgs = {
      url = "github:nixos/nixpkgs?ref=release-22.05";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
    };
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs {
            inherit system;
          };
        in
        rec {
          devShell = pkgs.mkShell
            {
              nativeBuildInputs = [ pkgs.bashInteractive ];
              buildInputs = with pkgs; [
                pkg-config
                openssl
                cargo-flamegraph
              ];
            };
        }
      );
}
