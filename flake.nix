{
  inputs.nixpkgs.url = "github:meta-introspector/nixpkgs?ref=feature/CRQ-016-nixify";
  outputs = { self, nixpkgs }:
    let pkgs = import nixpkgs { system = "x86_64-linux"; };
    in { devShells.x86_64-linux.default = pkgs.mkShell {
      buildInputs = [ pkgs.python3 pkgs.python3Packages.pip ];
    }; };
}
