# Working todo list!

In addition to scattered `# TODO:` comments in the code, we have some main todo
items (akin to a roadmap) that need to get done to get the project to a working
state.

# 10 nov 2026

Error correction is the current focus.

1. ~~Import the `galois` package (need to rename `galois.py`. maybe rename
   `galois.py` to `galoispn.py` for GF(p^n)).~~
2. ~~Implement RS/BCH ECC code in the error_correction package.~~
3. Update various functions in the error_correction package to use numpy instead of
   python's list augmentation.
4. Connect error_correction to modulate/demodulate.
5. Use requirements.txt, write simple devenv setup code in bash
6. ~~Write test code that takes wire data, corrupts a few bits, and decodes it.~~
7. Implement page interleaving
8. Import `pytest`! :D
9. Set up previous tests with `pytest`
10. Write a test harness for audio data
11. Write a simple audio distortion setup
12. Use distortion setup to distort data and do a full-integration test with pytest
    that takes test data -> encodes it -> modulates it -> distorts it with
    `distort.py` -> demodulates it -> decodes it -> compares it with test data
13. Write a todo list for next steps of this project :)
