"""Minimal reader for R .rda files holding data frames (XDR format, versions 2/3). Stdlib only."""
import bz2, csv, gzip, lzma, struct, sys


def opener(raw):
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)
    if raw[:3] == b"BZh":
        return bz2.decompress(raw)
    if raw[:6] == b"\xfd7zXZ\x00":
        return lzma.decompress(raw)
    return raw


class R:
    def __init__(self, b):
        assert b[:5] in (b"RDX2\n", b"RDX3\n"), b[:5]
        self.b, self.p, self.refs = b, 5, []
        assert self.b[self.p:self.p + 2] == b"X\n"
        self.p += 2
        ver = self.i()
        self.i(); self.i()
        if ver == 3:
            self.p += self.i()  # native encoding name

    def i(self):
        v = struct.unpack(">i", self.b[self.p:self.p + 4])[0]
        self.p += 4
        return v

    def item(self):
        flags = self.i()
        t, has_attr, has_tag = flags & 0xFF, flags & (1 << 9), flags & (1 << 10)
        if t in (254, 242, 241, 253, 252, 251, 250):
            return None
        if t == 255:  # REFSXP
            idx = flags >> 8
            return self.refs[(idx or self.i()) - 1]
        if t == 1:  # SYMSXP
            s = self.item()
            self.refs.append(s)
            return s
        if t in (2, 6):  # LISTSXP / LANGSXP -> list of (tag, value, attrs)
            out = []
            while True:
                attrs = self.item() if has_attr else None
                tag = self.item() if has_tag else None
                out.append((tag, self.item(), attrs))
                flags = self.i()
                t, has_attr, has_tag = flags & 0xFF, flags & (1 << 9), flags & (1 << 10)
                if t == 254:
                    return out
                assert t in (2, 6), t
        if t == 9:  # CHARSXP
            n = self.i()
            if n == -1:
                return None
            s = self.b[self.p:self.p + n].decode("utf-8", "replace")
            self.p += n
            return s
        if t == 238:  # ALTREP
            info, state, attr = self.item(), self.item(), self.item()
            cls = info[0][1]
            if cls == "compact_intseq":
                n, start, step = state
                val = [int(start + k * step) for k in range(int(n))]
            elif cls == "deferred_string":
                val = [str(x) for x in state[0][1]]
            elif cls.startswith("wrap_"):
                val = state[0]
            else:
                raise ValueError(cls)
            return Vec(val, dict((a[0], a[1]) for a in attr) if attr else {})
        if t in (10, 13, 14, 16, 19):
            n = self.i()
            if t in (10, 13):
                val = list(struct.unpack(f">{n}i", self.b[self.p:self.p + 4 * n]))
                self.p += 4 * n
                val = [None if v == -2147483648 else v for v in val]
            elif t == 14:
                val = list(struct.unpack(f">{n}d", self.b[self.p:self.p + 8 * n]))
                self.p += 8 * n
            else:
                val = [self.item() for _ in range(n)]
            attrs = self.item() if has_attr else []
            return Vec(val, {a[0]: a[1] for a in attrs})
        raise ValueError(f"unsupported SEXP type {t}")


class Vec(list):
    def __init__(self, val, attrs):
        super().__init__(val)
        self.attrs = attrs


def column(v):
    lv = v.attrs.get("levels")
    return [None if x is None else lv[x - 1] for x in v] if lv is not None else list(v)


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    objs = R(opener(open(src, "rb").read())).item()
    name, df, _ = objs[0]
    keep = [k for k, n in enumerate(df.attrs["names"]) if n != "input"]  # D8: input only repeats the previous choice
    names = [df.attrs["names"][k] for k in keep]
    cols = [column(df[k]) for k in keep]
    print("object:", name, "| class:", df.attrs.get("class"), "| rows:", len(cols[0]), "| cols:", list(names))
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        w.writerows(zip(*cols))
