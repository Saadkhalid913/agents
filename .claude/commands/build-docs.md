Build the documentation PDF.

Instructions:
1. Run `./docs/build.sh`
2. If the build succeeds, report the output file path and size
3. If the build fails, read the error output and diagnose the issue. Common problems:
   - Missing pandoc or xelatex: suggest `brew install pandoc` or installing TeX Live
   - Missing fonts: suggest installing Libertinus fonts
   - LaTeX errors: read the error message, check the markdown files for syntax issues
   - Missing metadata.yaml: check that `docs/metadata.yaml` exists
4. After a successful build, suggest the user open the PDF to verify formatting
