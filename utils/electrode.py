import re
from dataclasses import dataclass


PREFIX_VOCAB = [
    # frontopolar / frontal bands
    "Fp", "AF", "F", "FC",
    # central / parietal bands
    "C", "CP", "P", "PO",
    # occipital
    "O",
    # temporal & transitions
    "FT", "T", "TP",
    # midline-only in some montages (rare but seen)
    "I",  # inion-related (Oz below) sometimes I1/I2 etc.
]

PREFIX2ID = {p: i for i, p in enumerate(PREFIX_VOCAB)}  # 0..len-1
SIDE2ID = {"L": 0, "R": 1, "M": 2, "UNK": 3}

@dataclass
class Electrode:
    prefix: str
    side: str
    number: int

    style: str = "10-20"  # optional metadata about naming style, e.g. "10-20", "10-10", "custom"

    @property
    def prefix_id(self):
        return PREFIX2ID[self.prefix]
    
    @property
    def side_id(self):
        return SIDE2ID[self.side]
    
    def __str__(self):
        return f"{self.prefix}:{self.prefix_id}-{self.side}:{self.side_id}-{self.number}"
    
    def __repr__(self):
        return str(self)

    def __init__(self, name: str, style: str = "10-20"):
        n = name.strip()

        if style != "10-20":
            raise NotImplementedError(f"Only '10-20' style supported for now, got style={style!r}")

        # normalize z
        n = re.sub(r"Z$", "z", n)

        # normalize prefix casing a bit (keep 2-letter prefixes like AF/FC/CP/PO/TP/FT uppercase)
        # keep Fp as Fp
        if len(n) >= 2 and n[:2].lower() == "fp":
            n = "Fp" + n[2:]
        elif len(n) >= 2 and n[:2].upper() in {"AF", "FC", "CP", "PO", "TP", "FT"}:
            n = n[:2].upper() + n[2:]
        else:
            # single-letter prefixes: F/C/P/O/T/I -> uppercase first char
            n = n[0].upper() + n[1:]

        # apply aliases if exact match
        # n = ALIASES.get(n, n)

        # longest-prefix match from vocab to avoid "F" eating "Fp"/"FC"/...
        prefix = None
        for p in sorted(PREFIX_VOCAB, key=len, reverse=True):
            if n.startswith(p):
                prefix = p
                suffix = n[len(p):]
                break
        if prefix is None:
            raise ValueError(f"Unknown electrode prefix in name={name!r}")

        # suffix: "", "z", or digits
        if suffix == "z":
            side = "M"
            number = 0
        elif suffix == "":
            # allow bare prefix (rare / nonstandard). mark as unknown side/number
            side = "UNK"
            number = 0
        elif suffix.isdigit():
            number = int(suffix)
            side = "L" if (number % 2 == 1) else "R"
        else:
            raise ValueError(f"Invalid electrode suffix in name={name!r} (parsed suffix={suffix!r})")

        self.prefix = prefix
        self.side = side
        self.number = number
        self.style = style