#!/usr/bin/env python3
"""
ScamShield — Bước 1: gom dữ liệu về một schema chung và chia tập.

Schema gốc (mọi nguồn):  url | label | source | collected_at
    label: 1 = scam, 0 = safe
Cột thêm khi build:      url_norm | reg_domain | role
    role ∈ {train, val, T1, T3, benign_test}  -> mỗi role một file trong data/processed/

Cách dùng (các nguồn chỉ tải 1 lần; chạy lại sẽ bỏ qua nguồn đã có)
  python prepare_data.py download              # PhiUSIIL, Phishing.Database, PhishTank
  python prepare_data.py benign --n 3000       # Tranco + Common Crawl -> URL benign CÓ path (chạy lâu, resume được)
  python prepare_data.py build

OpenPhish và tập T2 do hệ thống riêng phụ trách, không nằm trong script này.

PhishVN tải tay (Mendeley -> Download All), giải nén vào data/raw/phishvn/
sao cho có data/raw/phishvn/splits/url_train.csv, url_test.csv ... và data/raw/phishvn/attacks/.

pip install ucimlrepo tldextract tranco pandas requests
"""
import argparse, datetime, hashlib, io, json, random, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import requests
import tldextract

RAW = Path("data/raw")
PROC = Path("data/processed")
PHISHVN = RAW / "phishvn"
TODAY = datetime.date.today().isoformat()
CORE = ["url", "label", "source", "collected_at"]

# suffix_list_urls=() -> dùng snapshot PSL đi kèm thư viện: kết quả ổn định, không cần mạng.
# include_psl_private_domains=True -> abc.pages.dev, xyz.web.app, foo.github.io được coi là
# domain riêng, không bị gộp chung một nhóm "pages.dev" khi split theo domain.
EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)

# ---- PhishVN: mở 1 file, xem tên cột thật rồi sửa ở đây nếu khác ----
PHISHVN_COLS = {
    "url": ["url", "URL"],
    "label": ["label", "is_phishing", "target", "class"],
    "collected_at": ["collected_at", "first_seen", "date", "timestamp"],
}
PHISHVN_KEEP = ["tier", "scenario", "impersonated_org", "label_source"]  # báo cáo theo tier/kịch bản; brand cho F4
PHISHVN_DROP = ["is_https"]                  # artefact thu thập (theo tác giả) – không bao giờ dùng


# =========================== Chuẩn hoá ===========================
def normalize(u):
    """lowercase host, bỏ scheme + fragment, giải mã punycode, bỏ 'www.', bỏ '/' cuối của trang chủ.
    Cột `url` gốc vẫn được giữ nguyên để F4 còn phát hiện được punycode/homoglyph."""
    if not isinstance(u, str):
        return None
    u = u.strip()
    if not u:
        return None
    if "://" not in u[:15]:
        u = "http://" + u
    try:
        p = urlsplit(u)
    except ValueError:
        return None
    host = p.netloc.lower()
    if not host:
        return None
    try:
        host = ".".join(l.encode("ascii").decode("idna") if l.startswith("xn--") else l
                        for l in host.split("."))
    except Exception:
        pass
    if host.startswith("www."):              # tránh shortcut "có www = benign" (Common Crawl hay có www)
        host = host[4:]
    path = "" if p.path in ("", "/") else p.path
    query = "?" + p.query if p.query else ""
    return host + path + query


def host_of(norm):
    h = norm.split("/", 1)[0].split("?", 1)[0]
    return h.rsplit("@", 1)[-1].split(":", 1)[0]


def reg_domain(norm):
    h = host_of(norm)
    e = EXTRACT(h)
    rd = getattr(e, "top_domain_under_public_suffix", None) or e.registered_domain
    return rd or h                            # IP hoặc host lạ -> dùng nguyên host làm nhóm


def bucket100(key):
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 100


# =========================== Download ===========================
def _try(name, fn):
    try:
        print(f"[ok]   {name}: {fn():,} dòng")
    except Exception as e:
        print(f"[lỗi] {name}: {e}")


def dl_phiusiil():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=967)
    df = pd.concat([ds.data.features[["URL"]], ds.data.targets], axis=1)
    df.columns = ["url", "label_raw"]
    out = pd.DataFrame({"url": df["url"], "label": 1 - df["label_raw"].astype(int),   # gốc 1 = legit
                        "source": "phiusiil", "collected_at": "2023"})
    out.to_csv(RAW / "phiusiil.csv", index=False)
    return len(out)


def dl_phishingdb():
    src = "https://github.com/mitchellkrogza/Phishing.Database/raw/master/phishing-links-ACTIVE.txt"
    r = requests.get(src, timeout=180)
    r.raise_for_status()
    urls = sorted({l.strip() for l in r.text.splitlines() if l.strip() and not l.startswith("#")})  # dedupe
    pd.DataFrame({"url": urls, "label": 1, "source": "phishingdb", "collected_at": TODAY}) \
        .to_csv(RAW / "phishingdb.csv", index=False)
    return len(urls)


def dl_phishtank():
    r = requests.get("http://data.phishtank.com/data/online-valid.csv.gz",
                     headers={"User-Agent": "phishtank/usth-scamshield"}, timeout=180)
    r.raise_for_status()
    try:
        df = pd.read_csv(io.BytesIO(r.content), compression="gzip")
    except Exception:
        df = pd.read_csv(io.BytesIO(r.content))
    t = pd.to_datetime(df["submission_time"], errors="coerce", utc=True).dt.date.astype(str)
    out = pd.DataFrame({"url": df["url"], "label": 1, "source": "phishtank", "collected_at": t})
    out.to_csv(RAW / "phishtank.csv", index=False)
    return len(out)


def cmd_download(a):
    """Tải 1 lần. Nguồn nào đã có file thì bỏ qua (dùng --force để tải lại)."""
    RAW.mkdir(parents=True, exist_ok=True)
    for name, fn, path in [("PhiUSIIL", dl_phiusiil, RAW / "phiusiil.csv"),
                           ("Phishing.Database", dl_phishingdb, RAW / "phishingdb.csv"),
                           ("PhishTank", dl_phishtank, RAW / "phishtank.csv")]:
        if path.exists() and not a.force:
            print(f"[có rồi] {name}: {path} -> bỏ qua")
            continue
        _try(name, fn)


# =========================== Benign có path ===========================
def cmd_benign(a):
    from tranco import Tranco
    RAW.mkdir(parents=True, exist_ok=True)
    saved = RAW / "tranco_top.txt"
    old = saved.read_text().split() if saved.exists() else []
    if len(old) >= a.top:
        # Tranco đổi mỗi ngày -> dùng lại đúng danh sách của lần chạy đầu để resume khớp
        # (kể cả khi chuyển từ Colab sang máy khác)
        top = old[: a.top]
        print(f"dùng lại danh sách Tranco đã lưu ({saved})")
    else:
        top = Tranco(cache=True, cache_dir=".tranco").list().top(a.top)
        saved.write_text("\n".join(top))
    vn = [d for d in top if d.endswith(".vn")]
    rest = [d for d in top if not d.endswith(".vn")][: a.n]
    domains = vn + rest
    print(f"{len(vn)} domain .vn + {len(rest)} domain khác")

    api = requests.get("https://index.commoncrawl.org/collinfo.json", timeout=60).json()[0]["cdx-api"]
    out_path, done_path = RAW / "benign_cc.csv", RAW / "benign_cc_done.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    todo = [d for d in domains if d not in done]
    print(f"còn {len(todo)} domain cần lấy ({len(done)} đã xong từ lần chạy trước), {a.workers} luồng")

    def fetch(d):
        urls = []
        for attempt in range(3):
            try:
                r = requests.get(api, params={"url": d, "matchType": "domain", "output": "json",
                                              "limit": 300, "fl": "url,status,mime"}, timeout=90)
                if r.status_code == 404:          # domain không có trong crawl
                    break
                if r.status_code in (429, 503):   # server quá tải -> chờ lâu dần rồi thử lại
                    time.sleep(10 * (attempt + 1))
                    continue
                r.raise_for_status()
                recs = [json.loads(l) for l in r.text.splitlines() if l.strip()]
                urls = [x["url"] for x in recs
                        if x.get("status") == "200" and "html" in x.get("mime", "")]
                break
            except Exception:
                time.sleep(5)
        time.sleep(a.sleep)                       # lịch sự với server Common Crawl
        urls = list(dict.fromkeys(urls))
        random.Random(d).shuffle(urls)            # CDX trả về theo thứ tự chữ cái -> xáo để đa dạng path
        return d, urls[: a.per_domain]

    # Ghi file chỉ ở luồng chính -> không bao giờ ghi chồng; mỗi domain xong là lưu ngay (resume được)
    n_urls = 0
    deadline = time.time() + a.max_minutes * 60 if a.max_minutes else None
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(fetch, d) for d in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            d, urls = fut.result()
            if urls:
                pd.DataFrame({"url": urls, "label": 0, "source": "commoncrawl", "collected_at": TODAY}) \
                    .to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
                n_urls += len(urls)
            with open(done_path, "a") as f:
                f.write(d + "\n")
            if i % 100 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} domain, +{n_urls:,} URL", flush=True)
            if deadline and time.time() > deadline:   # hết giờ: dừng gọn, phần còn lại để lượt sau
                for f_ in futs:
                    f_.cancel()
                print(f"  hết {a.max_minutes} phút -> dừng ở {i}/{len(todo)} domain, +{n_urls:,} URL", flush=True)
                break

    done = set(done_path.read_text().split()) if done_path.exists() else set()
    remaining = sum(d not in done for d in domains)
    (RAW / "benign_cc_remaining.txt").write_text(str(remaining))
    print("XONG toàn bộ." if remaining == 0 else f"Còn {remaining} domain: chạy lại đúng lệnh này để tiếp tục.")


# =========================== Build ===========================
def _pick(df, names, required=True):
    for n in names:
        if n in df.columns:
            return n
    if required:
        raise KeyError(f"Không thấy cột nào trong {names}. Cột hiện có: {list(df.columns)} "
                       f"-> sửa PHISHVN_COLS ở đầu file")
    return None


def load_phishvn(path, role, default_label=None):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.lower() in PHISHVN_DROP])
    c_url = _pick(df, PHISHVN_COLS["url"])
    c_lab = _pick(df, PHISHVN_COLS["label"], required=default_label is None)
    c_t = _pick(df, PHISHVN_COLS["collected_at"], required=False)
    out = pd.DataFrame({
        "url": df[c_url],
        "label": df[c_lab] if c_lab else default_label,
        "source": "phishvn",
        "collected_at": df[c_t].astype(str) if c_t else "2026-08",
    })
    if out["label"].dtype == object:
        out["label"] = out["label"].astype(str).str.lower().map(
            {"phishing": 1, "phish": 1, "scam": 1, "malicious": 1, "1": 1,
             "legitimate": 0, "legit": 0, "benign": 0, "safe": 0, "0": 0})
    bad = ~out["label"].isin([0, 1])
    if bad.any():
        print(f"  [cảnh báo] {path.name}: {bad.sum()} dòng nhãn lạ -> bỏ. Giá trị: "
              f"{df.loc[bad, c_lab].unique()[:5] if c_lab else '-'}")
        out = out[~bad]
    for c in df.columns:
        if any(k in c.lower() for k in PHISHVN_KEEP):
            out[c] = df.loc[out.index, c]
    if "source" in df.columns:               # nguồn gốc bên trong PhishVN (giữ lại, không ghi đè cột source)
        out["phishvn_source"] = df.loc[out.index, "source"]
    out["role"] = role
    print(f"  phishvn/{path.parent.name}/{path.name} -> {role}: {len(out):,} dòng, "
          f"nhãn {out['label'].value_counts().to_dict()}")
    return out


def _read(path, role, phishing_only=False, cap=0):
    # on_bad_lines="skip": nếu lần chạy benign bị dừng đúng lúc đang ghi, dòng cuối có thể dở dang
    df = pd.read_csv(path, usecols=CORE, on_bad_lines="skip")
    if phishing_only:
        df = df[df["label"] == 1]
    n = len(df)
    if cap and n > cap:   # một nguồn quá lớn sẽ lấn át PhishVN -> lấy mẫu ngẫu nhiên cố định seed
        df = df.sample(cap, random_state=42)
    df["role"] = role
    print(f"  {path.name} -> {role}: {len(df):,} dòng" + (f" (lấy mẫu từ {n:,})" if len(df) < n else ""))
    return df


def cmd_build(a):
    split_dir = PHISHVN / "splits"
    if not split_dir.exists():
        sys.exit("Chưa có data/raw/phishvn/splits/ — PhishVN là nguồn bắt buộc, tải và giải nén trước.")
    frames = []

    # 1) PhishVN: giữ nguyên split của tác giả (đã chia theo domain + thời gian)
    for f in sorted(split_dir.glob("url_*.csv")):
        n = f.stem.lower()
        role = "T1" if "test" in n else "val" if "val" in n else "train" if "train" in n else None
        if role:
            frames.append(load_phishvn(f, role))
    for f in sorted((PHISHVN / "attacks").rglob("*.csv")):
        if "url" not in pd.read_csv(f, nrows=0).columns:
            # PhishVN v4: attacks/ là biến thể TIN NHẮN (cột text), không phải URL -> không dùng cho model URL
            print(f"  bỏ qua attacks/{f.name}: không có cột url (biến thể tin nhắn)")
            continue
        frames.append(load_phishvn(f, "T3", default_label=1))

    # 2) Phishing quốc tế -> "pool" (chia train/val theo domain bên dưới)
    if (RAW / "phiusiil.csv").exists():
        # Mặc định chỉ lấy phần phishing: benign của PhiUSIIL phần lớn là trang chủ (không path)
        frames.append(_read(RAW / "phiusiil.csv", "pool", phishing_only=not a.phiusiil_benign, cap=a.cap))
    for name in ["phishingdb.csv", "phishtank.csv"]:
        if (RAW / name).exists():
            frames.append(_read(RAW / name, "pool", cap=a.cap))

    # 3) Benign có path
    if (RAW / "benign_cc.csv").exists():
        frames.append(_read(RAW / "benign_cc.csv", "benign"))
    else:
        print("  [cảnh báo] chưa có benign_cc.csv — train sẽ chỉ có benign của PhishVN")

    df = pd.concat(frames, ignore_index=True)
    df["url_norm"] = df["url"].map(normalize)
    df = df.dropna(subset=["url_norm"]).copy()
    df["label"] = df["label"].astype(int)
    df["reg_domain"] = df["url_norm"].map(reg_domain)

    # 4) Chia pool/benign THEO DOMAIN (một domain chỉ ở một tập)
    b = df["reg_domain"].map(bucket100)
    m = df["role"] == "pool"
    df.loc[m, "role"] = b[m].map(lambda x: "val" if x < 10 else "train")
    m = df["role"] == "benign"
    df.loc[m, "role"] = b[m].map(lambda x: "val" if x < 10 else "benign_test" if x < 25 else "train")

    # 5) Chống rò rỉ: domain đã ở val/test thì không được ở train (trừ nền tảng lớn top-1000 Tranco
    #    như google.com, vì sites.google.com/... vừa có benign vừa bị lạm dụng)
    top = RAW / "tranco_top.txt"
    exempt = set(top.read_text().split()[:1000]) if top.exists() else set()

    def purge(frm, against, by_domain=True):
        nonlocal df
        ref = df[df["role"].isin(against)]
        bad_d = set(ref["reg_domain"]) - exempt if by_domain else set()
        bad_u = set(ref["url_norm"])
        hit = df["role"].isin(frm) & (df["reg_domain"].isin(bad_d) | df["url_norm"].isin(bad_u))
        if hit.any():
            print(f"  purge {frm} theo {against}: bỏ {hit.sum():,} dòng")
        df = df[~hit]

    purge(["train", "val"], ["T1", "benign_test"])
    purge(["train", "val"], ["T3"], by_domain=False)     # T3 là biến thể né tránh: chỉ bỏ URL trùng
    purge(["train"], ["val"])

    # 6) Dedupe trong từng tập; URL trùng mà mâu thuẫn nhãn thì bỏ hết
    nun = df.groupby(["role", "url_norm"])["label"].transform("nunique")
    print(f"  bỏ {(nun > 1).sum():,} dòng mâu thuẫn nhãn")
    before = (nun == 1).sum()
    df = df[nun == 1].drop_duplicates(["role", "url_norm"])   # giữ bản đầu: PhishVN đứng trước nên giữ được tier
    print(f"  dedupe: bỏ {before - len(df):,} dòng trùng")

    # 7) Ghi file
    PROC.mkdir(parents=True, exist_ok=True)
    names = {"train": "train", "val": "val", "T1": "test_T1",
             "T3": "test_T3", "benign_test": "benign_test"}
    for name in names.values():          # xoá kết quả cũ để không còn file của lần build trước
        (PROC / f"{name}.csv").unlink(missing_ok=True)
    for role, name in names.items():
        part = df[df["role"] == role]
        if len(part):
            part.to_csv(PROC / f"{name}.csv", index=False)
    stats = pd.crosstab([df["role"], df["source"]], df["label"]).rename(columns={0: "safe", 1: "scam"})
    stats.to_csv(PROC / "stats.csv")
    print("\n=== Thống kê (dán vào data card) ===")
    print(stats.to_string())

    # 8) Kiểm tra shortcut "URL dài / có path = scam"
    tr = df[df["role"] == "train"].assign(n=lambda x: x["url_norm"].str.len(),
                                          has_path=lambda x: x["url_norm"].str.contains(r"[/?]"))
    print("\n=== Kiểm tra shortcut trên train ===")
    shortcut = tr.groupby("label").agg(so_dong=("n", "size"), do_dai_trung_vi=("n", "median"),
                                       ti_le_co_path=("has_path", "mean")).round(3)
    print(shortcut.to_string())
    print("Nếu safe có tỉ lệ path thấp hơn hẳn scam -> cần thêm benign có path trước khi train.")

    # 9) Data card tự sinh (bổ sung phần mô tả bằng tay sau)
    raw_dates = {p.name: datetime.date.fromtimestamp(p.stat().st_mtime).isoformat()
                 for p in sorted(RAW.glob("*.csv")) if p.name != "benign_cc_done.txt"}
    card = [f"# ScamShield — Data card (tự sinh {TODAY})", "",
            f"- Giới hạn mỗi nguồn quốc tế: {a.cap:,} URL" if a.cap else "- Không giới hạn mỗi nguồn",
            "- Nhãn: 1 = scam, 0 = safe. Split theo registered domain (một domain chỉ ở một tập).",
            "- Đã bỏ: cột is_https của PhishVN; benign của PhiUSIIL; scheme và www. trong url_norm.", "",
            "## Ngày tải dữ liệu gốc", "", *[f"- `{k}`: {v}" for k, v in raw_dates.items()], "",
            "## Số lượng theo tập × nguồn", "", "```", stats.to_string(), "```", "",
            "## Kiểm tra shortcut trên train", "", "```", shortcut.to_string(), "```", "",
            "## Ý nghĩa từng file", "",
            "| File | Dùng để |", "|---|---|",
            "| train.csv | huấn luyện |",
            "| val.csv | chọn ngưỡng, tuning |",
            "| test_T1.csv | PhishVN test: bối cảnh VN, chia theo thời gian |",
            "| test_T3.csv | URL né tránh (PhishVN v4 không có; nhóm red-team tạo ở tuần 9–10) |",
            "| benign_test.csv | benign của domain chưa thấy khi train |"]
    (PROC / "DATA_CARD.md").write_text("\n".join(card), encoding="utf-8")
    print(f"\nĐã ghi {PROC / 'DATA_CARD.md'}")


# =========================== CLI ===========================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("download")
    s.add_argument("--force", action="store_true", help="tải lại cả nguồn đã có")
    s.set_defaults(fn=cmd_download)
    s = sub.add_parser("benign")
    s.add_argument("--top", type=int, default=100_000, help="lấy Tranco top-N để lọc .vn")
    s.add_argument("--n", type=int, default=3000, help="số domain không phải .vn")
    s.add_argument("--per-domain", type=int, default=30)
    s.add_argument("--sleep", type=float, default=1.0)
    s.add_argument("--workers", type=int, default=3, help="số luồng song song (đừng để quá 4)")
    s.add_argument("--max-minutes", type=float, default=0,
                   help="tự dừng gọn sau N phút (dùng cho GitHub Actions); 0 = chạy đến hết")
    s.set_defaults(fn=cmd_benign)
    s = sub.add_parser("build")
    s.add_argument("--phiusiil-benign", action="store_true", help="dùng cả benign của PhiUSIIL (không khuyến nghị)")
    s.add_argument("--cap", type=int, default=100_000,
                   help="tối đa N URL mỗi nguồn quốc tế (PhiUSIIL, Phishing.Database, PhishTank); 0 = không giới hạn")
    s.set_defaults(fn=cmd_build)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
