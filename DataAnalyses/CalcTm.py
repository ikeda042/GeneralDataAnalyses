import numpy as np


def Tm(seq: str) -> float:
    parameters: dict[str, dict[str, float]] = {
        "deltaH": {
            "AA": -9.1,
            "AT": -8.6,
            "TA": -6.0,
            "CA": -5.8,
            "GT": -6.5,
            "CT": -7.8,
            "GA": -5.6,
            "CG": -11.9,
            "GC": -11.1,
            "GG": -11.0,
            "TT": -9.1,
            "AT": -8.6,
            "TA": -6.0,
            "TG": -5.8,
            "AC": -6.5,
            "AG": -7.8,
            "TC": -5.6,
            "CG": -11.9,
            "GC": -11.1,
            "CC": -11.0,
        },
        "deltaS": {
            "AA": -0.024,
            "AT": -0.0239,
            "TA": -0.0169,
            "CA": -0.0129,
            "GT": -0.0173,
            "CT": -0.0208,
            "GA": -0.0135,
            "CG": -0.0278,
            "GC": -0.0267,
            "GG": -0.0266,
            "TT": -0.024,
            "AT": -0.0239,
            "TA": -0.0169,
            "TG": -0.0129,
            "AC": -0.0173,
            "AG": -0.0208,
            "TC": -0.0135,
            "CG": -0.0278,
            "GC": -0.0267,
            "CC": -0.0266,
        },
    }
    deltaH, deltaS = 0, 0
    for i in range(len(seq) - 1):
        deltaH += parameters["deltaH"][seq[i : i + 2]]
        deltaS += parameters["deltaS"][seq[i : i + 2]]
    return (
        (deltaH) / (-0.0108 + deltaS + 0.00199 * np.log(0.0000005 / 4))
        - 273.15
        + 16.6 * np.log10(0.05)
    )


seq = """CGATAAGCTAGCTTCACGCTGCCGCAAGCACTCAGGGCGCAAGGGCTGCTAAAGGAAGCGGAACACGTAGAAAGCCAGTCCGCAGAAACGGTGCTGACCCCGGATGAATGTCAGCTACTGGGCTATCTGGACAAGGGAAAACGCAAGCGCAAAGAGAAAGCAGGTAGCTTGCAGTGGGCTTACATGGCGATAGCTAGACTGGGCGGTTTTATGGACAGCAAGCGAACCGGAATTGCCAGCTGGGGCGCCCTCTGGTAAGGTTGGGAAGCCCTGCAAAGTAAACTGGATGGCTTTCTTGCCGCCAAGGATCTGATGGCGCAGGGGATCAAGATCTGATCAAGAGACAGGATGAGGATCGTTTCGCATGATTGAACAAGATGGATTGCACGCAGGTTCTCCGGCCGCTTGGGTGGAGAGGCTATTCGGCTATGACTGGGCACAACAGACAATCGGCTGCTCTGATGCCGCCGTGTTCCGGCTGTCAGCGCAGGGGCGCCCGGTTCTTTTTGTCAAGACCGACCTGTCCGGTGCCCTGAATGAACTCCAAGACGAGGCAGCGCGGCTATCGTGGCTGGCCACGACGGGCGTTCCTTGCGCAGCTGTGCTCGACGTTGTCACTGAAGCGGGAAGGGACTGGCTGCTATTGGGCGAAGTGCCGGGGCAGGATCTCCTGTCATCTCACCTTGCTCCTGCCGAGAAAGTATCCATCATGGCTGATGCAATGCGGCGGCTGCATACGCTTGATCCGGCTACCTGCCCATTCGACCACCAAGCGAAACATCGCATCGAGCGAGCACGTACTCGGATGGAAGCCGGTCTTGTCGATCAGGATGATCTGGACGAAGAGCATCAGGGGCTCGCGCCAGCCGAACTGTTCGCCAGGCTCAAGGCGCGGATGCCCGACGGCGAGGATCTCGTCGTGACCCATGGCGATGCCTGCTTGCCGAATATCATGGTGGAAAATGGCCGCTTTTCTGGATTCATCGACTGTGGCCGGCTGGGTGTGGCGGACCGCTATCAGGACATAGCGTTGGCTACCCGTGATATTGCTGAAGAGCTTGGCGGCGAATGGGCTGACCGCTTCCTCGTGCTTTACGGTATCGCCGCTCCCGATTCGCAGCGCATCGCCTTCTATCGCCTTCTTGACGAGTTCTTCTGAGCGGGACTCTGGGGTTCGAAATGACCGACCAAGCGACGCCCAACCTGCCATCACGAGATTTCGATTCCACCGCCGCCTTCTATGAAAGGTTGGGCTTCGGAATCGTTTTCCGGGACGCCCTCGCGGACGTGCTCATAGTCCACGACGCCCGTGATTTTGTAGCCCTGGCCGACGGCCAGCAGGTAGGCCGACAGGCTCATGCCGGCCGCCGCCGCCTTTTCCTCAATCGCTCTTCGTTCGTCTGGAAGGCAGTACACCTTGATAGGTGGGCTGCCCTTCCTGGTTGGCTTGGTTTCATCAGCCATCCGCTTGCCCTCATCTGTTACGCCGGCGGTAGCCGGCCAGCCTCGCAGAGCAGGATTCCCGTTGAGCACCGCCAGGTGCGAATAAGGGACAGTGAAGAAGGAACACCCGCTCGCGGGTGGGCCTACTTCACCTATCCTGCCCCGCTGACGCCGTTGGATACACCAAGGAAAGTCTACACGAACCCTTTGGCAAAATCCTGTATATCGTGCGAAAAAGGATGGATATACCGAAAAAATCGCTATAATGACCCCGAAGCAGGGTTATGCAGCGGAAAAGCGCTGCTTCCCTGCTGTTTTGTGGAATATCTACCGACTGGAAACAGGCAAATGCAGGAAATTACTGAACTGAGGGGACAGGCGAGAGACGATGCCAAAGAGCTCCTGAAAATCTCGATAACTCAAAAAATACGCCCGGTAGTGATCTTATTTCATTATGGTGAAAGTTGGAACCTCTTACGTGCCGATCAACGTCTCATTTTCGCCAAAAGTTGGCCCAGGGCTTCCCGGTATCAACAGGGACACCAGGATTTATTTATTCTGCGAAGTGATCTTCCGTCACAGGTATTTATTCGGCGCAAAGTGCGTCGGGTGATGCTGCCAACTTACTGATTTAGTGTATGATGGTGTTTTTGAGGTGCTCCAGTGGCTTCTGTTTCTATCAGCTCCTGAAAATCTCGATAACTCAAAAAATACGCCCGGTAGTGATCTTATTTCATTATGGTGAAAGTTGGAACCTCTTACGTGCCGATCAACGTCTCATTTTCGCCAAAAGTTGGCCCAGGGCTTCCCGGTATCAACAGGGACACCAGGATTTATTTATTCTGCGAAGTGATCTTCCGTCACAGGTATTTATTCGGCGCAAAGTGCGTCGGGTGATGCTGCCAACTTACTGATTTAGTGTATGATGGTGTTTTTGAGGTGCTCCAGTGGCTTCTGTTTCTATCAGGGCTGGATGATCCTCCAGCGCGGGGATCTCATGCTGGAGTTCTTCGCCCACCCCAAAAGGATCTAGGTGAAGATCCTTTTTGATAATCTCATGACCAAAATCCCTTAACGTGAGTTTTCGTTCCACTGAGCGTCAGACCCCGTAGAAAAGATCAAAGGATCTTCTTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTTGCAAACAAAAAAACCACCGCTACCAGCGGTGGTTTGTTTGCCGGATCAAGAGCTACCAACTCTTTTTCCGAAGGTAACTGGCTTCAGCAGAGCGCAGATACCAAATACTGTCCTTCTAGTGTAGCCGTAGTTAGGCCACCACTTCAAGAACTCTGTAGCACCGCCTACATACCTCGCTCTGCTAATCCTGTTACCAGTGGCTGCTGCCAGTGGCGATAAGTCGTGTCTTACCGGGTTGGACTCAAGACGATAGTTACCGGATAAGGCGCAGCGGTCGGGCTGAACGGGGGGTTCGTGCACACAGCCCAGCTTGGAGCGAACGACCTACACCGAACTGAGATACCTACAGCGTGAGCATTGAGAAAGCGCCACGCTTCCCGAAGGGAGAAAGGCGGACAGGTATCCGGTAAGCGGCAGGGTCGGAACAGGAGAGCGCACGAGGGAGCTTCCAGGGGGAAACGCCTGGTATCTTTATAGTCCTGTCGGGTTTCGCCACCTCTGACTTGAGCGTCGATTTTTGTGATGCTCGTCAGGGGGGCGGAGCCTATGGAAAAACGCCAGCAACGCGGCCTTTTTACGGTTCCTGGCCTTTTGCTGGCCTTTTGCTCACATGTTCTTTCCTGCGTTATCCCCTGATTCTGTGGATAACCGTATTACCGCCTTTGAGTGAGCTGATACCGCTCGCCGCAGCCGAACGACCGAGCGCAGCGAGTCAGTGAGCGAGGAAGCGGAAGAGCGCCCAATACGCAAACCGCCTCTCCCCGCGCGTTGGCCGATTCATTAATGCAGCTGGCACGACAGGTTTCCCGACTGGAAAGCGGGCAGTGAGCGCAACGCAATTAATGTGAGTTAGCTCACTCATTAGGCACCCCAGGCTTTACACTTTATGCTTCCGGCTCGTATGTTGTGTGGAATTGTGAGCGGATAACAATTTCACACAGGAAACAGCTATGACCATGATTACGAATTCGAGCTCGGTACCCtgacgttctagACCCTACCTCTCCAAGATTACCGGCGTGCCCATGGTGGCCCTGGCCACCAACGTCATGCTGGGCAAAAGCCTGCCGGAGCAGGGCTACCGGGGCGGCTTAATGCCGCCGCCGGATTTTACCGCCGTAAAGGTCCCCGTCTTTTCCTTCGGCAAGCTGTTGCAGGTGGACACCTCCCTGGGACCGGAGATGAAGTCCACCGGCGAGGTAATGGGGATTGATCCCGTCTTCGAACGCGCCCTCTATAAAGGCCTGGTAGCCGCCGGCTGCTCCATCCCCCATCACGGCACCCTGCTGGCGACCATCGCCGATAAGGACAAGGCGGAAGCAGTGCCCATCATCAAGGGCTTTGCCGAACTGGGCTTCCAGGTGGTGGCTACCGCCGGCACCGCCGGCGCCCTGGCCGCAGCGGGACTCTTCGTAGAGAGGGTGGGGAAGATCCGCGAGGGTTCGCCCCACATTATCGACTATATCCGGGAAGGGAAGGTCCACTTTGTCCTCAACACCCTCACCAGGGGCAAGATGCCCGGCCGGGACGGTTTTAAGATCCGCCGCGCCGCGGCCGAACTGGGCATCCCCTGCCTGACTTCCCTGGATACGGCCCGGGCCCTGCTCAAAGTCCTCCAGTCCCTGAAGTCCGGCGACGGGTTTAACCTCAAACCCCTGCAGGAGTATGTACCCCTTTCCCGTCCTTAACGGAGGAGCGCCAAATCGCCTCCGCCCCACCCCGGCAGGAGCAGCAGCCCGCGGCTGCACCGGCCGGGCGGTTTCCCGGCCGGCCCTTCAAGCACCAGGCGAGATGGCCGGGCCGCCGCCATTTAGCATATCAAGAGCGCCGGAAGGGAAGGGCTTTTCCGGTTTTTACCGGTCGGGGTTAAGCCTGACTTAAGGGCCGGTACCGGACCCTCCCCATATTCACTCCGCTTACACTCCGTTTTTTGAACTATAAGATCATAAAGCGATATTTAAGGGCTTCTGGCCTGCTTGCCAACACTAATGTACCTGCAGGAGATGATCCGCATGCATGCCAAGGACAAAATAATCGTCGCCCTGGATGTTCCCGACCTGGCTGCCGGGGAAAAGCTGGTGGACCGGCTTTCCCCCTACGCCGGCATGTTTAAAGTCGGCCTGGAGTTTTTCACCGCCGCCGGGCCGGCGGCCGTCCGGATGGTAAAGGAGCGTGGTGGCCGGGTATTTGCCGACCTGAAGTTCCACGACATCCCCAACACCGTGGCCGGAGCGGCGCGGGCCCTGGTGCGCCTGGGCGTGGATATGCTCAACGTTCACGCCGCCGGCGGCAAGGCCATGCTGCAGGCTGCCGCCGCCGCCGTCCGGGAGGAGGCCGCGGCCTTAAACCGCCCGGCGCCGGTAATAATCGCGGTCACTGTTTTGACCAGCCTGGACAGGGAAGCTCTACGCTGCGAGGTGGGTATCGAGCGAGAGGTAGAAGAACAGGTGGCCCGCTGGGCGCTCCTGGCCCGGGAGGCCGGCCTGGACGGCGTAGTAGCCTCGCCCCGGGAGATCCGGGCCATCCGGGAGGCCTGCGGGCCGGAGTTCGTCATCGTGACCCCGGGCGTGCGCCCGGCTGGGTCCGACCGGGGCGACCAGCGCCGGGTCATGACCCCGGCCGAGGCCCTGCGGGAGGGCGCCTCCTACCTGGTCATCGGCCGGCCCATCACCGCGGCCCCCGACCCCGTCGCCGCCGCCCGGGCCATCGCGGCGGAAATAGAGATGGTGAAATAATAACTGGACGGTTGCCAAGTACCGGGACGAGCAGGGCATCCCGGCGGCGGCTAAAAGAAAACGATATTAGTTAAGAAGGATTTTGACCATTTGTGTTGAATAGATAGTGTTTGACGGTACAATCTCCGGCAATTAGCAATATATCATAATAAATCCTGATTGGGTTAGGAATAATATCAAAAGCCAAGGAGCCTGAAAGCGGTGGGGGTTGACGCTGCAGGAATTTAACCCTTGCCGTTACAATAAATATAAGGAGGAGTACATAATGAACTTCAACAAGATCGATCTGGACAACTGGAAACGCAAGGAAATCTTCAACCATTATCTGAACCAGCAGACCACCTTCTCCATCACCACGGAGATCGACATCTCCGTGCTGTACCGGAACATCAAGCAAGAAGGCTACAAGTTCTACCCCGCCTTCATCTTTCTGGTCACGCGGGTCATCAACAGCAACACCGCCTTCCGCACCGGGTACAACTCCGACGGCGAGCTCGGCTACTGGGACAAGCTGGAACCGCTCTACACCATCTTCGACGGCGTGAGCAAGACCTTTAGCGGCATCTGGACGCCCGTGAAGAACGACTTCAAGGAGTTCTACGATCTGTACCTCTCCGACGTGGAAAAGTACAACGGCTCCGGCAAGCTGTTCCCGAAAACCCCCATTCCCGAAAACGCCTTTTCCCTCTCCATCATCCCGTGGACCAGCTTCACCGGCTTTAATCTGAACATTAACAACAACAGCAACTATCTGCTGCCGATCATCACCGCCGGCAAGTTCATCAACAAGGGCAACAGCATCTACCTCCCGCTGAGTCTGCAAGTCCACCACAGCGTCTGCGATGGCTACCACGCCGGGCTGTTTATGAACAGCATCCAAGAACTGAGCGACCGCCCGAATGACTGGCTGCTGTAATCGGCCTGCTTTCATGCTTGATAATTTTTGTCATGTAGGGCTACAATGATAGTAACAGGTGATGACACGATGGAACGAATTAACTTTATCAATACCCGGGAGTTTAAAAATAGAGCAACCCAAATCTTGAGGCAGGTACAAAAAGACCAGGTTATTATTATAACCAATCGCGGTAAACCTGTAGCCACTTTAAAAGGTTTCAATCCACGTGACCTGGTTGTTGCAGAAGATAGACATGATAGCCTTTACCAGCATTTGCGGCAACAAATTTTAAAAGAAAGTCCAGAACTGGCTGCCAGGGATACCAGGCAAATCGCCACTGATTTTGAAAAGATAACAGCTAAAATGAGAAAACAGATTGCCTACAGGACCTGGGAAGAAATGGACCGGCACTTAAAAGGGGATCCTTATGATCTTACTGGATACTAATATTTTTATAATCGATCGTTTTTTTCCAGGGGATAGTCATTACGCTATAAACAAGGAATTCATTCAAGAGTTATCGCGGCTCGAGGCGGGGTTTTCTATTTTTTCGTTATTAGAACTTACCGGCATTGCTTCTTTTAACCTTTCAGCCAAAGAATTGCAGCAATGGTTGTTTGATTTTGCCTCCGTTTATCCTATTCGTATTCTTGATCCCTATGATTTAAAGATTGATTCTGCCAAGGAATGGTATACTAAATTTTTGCAGGAACTAATGGCAAAAGTTACCCACCAAATGACTTTTGGCGACGCTATTTTTTTACGTGAAGCTGAAGGTTATCAAGTAGAGTATATTATTAGCTGGAATAAGAAACATTTTCTTTCACGTACGACAATCAAAGTGCTCAACCCTGAAGAGTTTTTGACAATATGGAAACCGCAATAATTCCTTTGGTCATAAGCAAGGCGTGGGCTTTTTTAATCTCGTTGGAATGGGTAACTTTATGGGGTTAGCTCCCGGCAACCTAAATCGGAAGGTGCATAAGCTTGGACagatatcaggtcaGGGGATCCTCTAGAGTCGACCTGCAGGCATGCAAGCTTGGCACTGGCCGTCGTTTTACAACGTCGTGACTGGGAAAACCCTGGCGTTACCCAACTTAATCGCCTTGCAGCACATCCCCCTTTCGCCAGCTGGCGTAATAGCGAAGAGGCCCGCACCGATCGCCCTTCCCAACAGTTGCGCAGCCTGAATGGCGAATGGCG
""".upper()


def get_GC_rate(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq)


"""
Constraints:
1. 25 <= len(seq) <= 35
2. 0.45 <= get_GC_rate(seq) <= 0.6
3. Tm(seq) >= 63
4. not AT-rich
5. GC3' end
"""

primer_candidates: list[str] = [
    seq[i : bp + i]
    for bp in range(25, 36)
    for i in range(0, len(seq) - bp)
    if 0.45 < get_GC_rate(seq[i : bp + i]) < 0.6
    and Tm(seq[i : bp + i]) > 63
    and (
        (seq[i : bp + i][0] == "G" and seq[i : bp + i][-1] == "C")
        or (seq[i : bp + i][0] == "C" and seq[i : bp + i][-1] == "G")
    )
    and "AAAA" not in seq[i : bp + i]
    and "TTTT" not in seq[i : bp + i]
]

print(primer_candidates)
