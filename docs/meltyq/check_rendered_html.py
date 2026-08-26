"""Check semantic links and identifiers in the rendered documentation."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


SOURCE_URL = (
    "https://github.com/HajimeKawahara/exoexamples/"
    "blob/main/examples/meltyq/meltyq_figure8.py"
)
EXOPIE_URL = "https://github.com/mplotnyko/exopie/"
EXOPIE_PAPER_URL = "https://doi.org/10.1093/mnras/stae993"
FIGURE3_SOURCE_URLS = {
    "https://github.com/HajimeKawahara/exoexamples/"
    "blob/main/examples/meltyq/meltyq_figure3.py",
    "https://github.com/HajimeKawahara/exoexamples/"
    "blob/main/examples/meltyq/meltyq_figure3_build_diffgrid.py",
}
FIGURE3_TAUREX_URLS = {
    "https://github.com/ucl-exoplanets/taurex3/blob/"
    "7b6e82a86d4675f140e9e59f3d1410a863251c03/"
    "src/taurex/data/planet.py",
    "https://github.com/ucl-exoplanets/taurex3/blob/"
    "7b6e82a86d4675f140e9e59f3d1410a863251c03/"
    "src/taurex/model/transmission.py",
    "https://github.com/ucl-exoplanets/taurex3/blob/"
    "7b6e82a86d4675f140e9e59f3d1410a863251c03/"
    "src/taurex/model/simplemodel.py",
    "https://github.com/ucl-exoplanets/taurex3/blob/"
    "7b6e82a86d4675f140e9e59f3d1410a863251c03/"
    "src/taurex/cia/hitrancia.py",
    "https://github.com/ucl-exoplanets/taurex3/blob/"
    "7b6e82a86d4675f140e9e59f3d1410a863251c03/"
    "src/taurex/util/scattering.py",
    "https://github.com/groningen-exoatmospheres/taurex-pymiescatt/"
    "commit/2973acec3985c2222281062be16a07428c43d621",
}
FIGURE3_EXPECTED_CODE = {
    "OpaCIA.logacia_matrix",
    "apply_bin_operator",
    "band_mean_bin_operator",
    "compare_diffgrid_with_teacher",
    "diffgrid_interval_midpoint_temperatures",
    "exojax.atm.idealgas.number_density",
    "exojax_simpson",
    "hydrostatic_radius_profile_ideal_gas",
    "layer_optical_depth_from_cross_section",
    "layer_optical_depth_from_extinction",
    "layer_optical_depth_from_log_cia",
    'nx_even_from_resolution_eslog(..., definition="pointwise")',
    "piecewise_linear_bin_operator",
    'resolution_eslog(..., definition="pointwise")',
    "taurex_rectangle",
}
EXPECTED_CODE = {
    "boundary.model_state.melt_volatile_mole_ratios",
    "exogibbs.api.magma_gas.solve",
    "exogibbs.interop.exoeos.make_pure_lnphi_func",
    "gas.solve_profile",
    "hydrostatic_radius_profile",
    "ln_n2_dasgupta2022",
    "MeltyqMagmaGasModel.evaluate",
    "pressure_layer_logspace_from_boundaries",
}
BAD_VISIBLE_PATTERNS = {
    "split pressure-grid identifier": r"pressure_layer_logspa\s+ce_from_boundaries",
    "split hydrostatic identifier": r"hydrost\s+atic_radius_profile",
    "raw RST link markup": r"meltyq_figure8\.py\s*<ht\s*tps?://",
    "raw math role": r":math:`",
    "misleading R_core output label": r"R_core=",
    "misleading Fortney-core label": r"Fortney-core",
    "misleading ExoPie-core label": r"ExoPie core",
}


class RenderedDocumentParser(HTMLParser):
    """Collect visible text, code literals, links, and image sources."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.visible_parts: list[str] = []
        self.code_values: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.image_sources: list[str] = []
        self.cite_count = 0
        self._code_depth = 0
        self._code_parts: list[str] = []
        self._link_href: str | None = None
        self._link_parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag in {"script", "style"}:
            self._ignored_depth += 1
        if tag == "code":
            self._code_depth += 1
            if self._code_depth == 1:
                self._code_parts = []
        if tag == "cite":
            self.cite_count += 1
        if tag == "a":
            self._link_href = attributes.get("href")
            self._link_parts = []
        if tag == "img" and attributes.get("src"):
            self.image_sources.append(attributes["src"] or "")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
        if tag == "code" and self._code_depth:
            self._code_depth -= 1
            if self._code_depth == 0:
                self.code_values.append("".join(self._code_parts).strip())
        if tag == "a" and self._link_href is not None:
            self.links.append(
                (self._link_href, "".join(self._link_parts).strip())
            )
            self._link_href = None
            self._link_parts = []

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.visible_parts.append(data)
        if self._code_depth:
            self._code_parts.append(data)
        if self._link_href is not None:
            self._link_parts.append(data)


def _local_target_exists(page: Path, target: str) -> bool:
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return True
    path = unquote(parsed.path)
    return (page.parent / path).resolve().exists()


def check_page(page: Path) -> list[str]:
    parser = RenderedDocumentParser()
    parser.feed(page.read_text(encoding="utf-8"))
    errors: list[str] = []

    missing_code = sorted(EXPECTED_CODE.difference(parser.code_values))
    if missing_code:
        errors.append(f"missing code literals: {missing_code}")
    split_code = sorted(
        {
            value
            for value in parser.code_values
            if re.sub(r"\s+", "", value) in EXPECTED_CODE
            and value != re.sub(r"\s+", "", value)
        }
    )
    if split_code:
        errors.append(f"split code literals: {split_code}")
    if parser.cite_count:
        errors.append(f"unexpected cite elements: {parser.cite_count}")

    visible_text = " ".join(" ".join(parser.visible_parts).split())
    for label, pattern in BAD_VISIBLE_PATTERNS.items():
        if re.search(pattern, visible_text):
            errors.append(label)

    source_links = [
        text
        for href, text in parser.links
        if href == SOURCE_URL and text == "meltyq_figure8.py"
    ]
    if not source_links:
        errors.append("missing semantic meltyq_figure8.py source link")

    link_targets = {href for href, _ in parser.links}
    for expected in (EXOPIE_URL, EXOPIE_PAPER_URL):
        if expected not in link_targets:
            errors.append(f"missing ExoPie provenance link: {expected}")
    external_statement = (
        "ExoFamilyの構成packageではない"
        if page.name.endswith("_ja.html")
        else "not part of ExoFamily"
    )
    if external_statement not in visible_text:
        errors.append("missing explicit external-ExoPie statement")

    for target in [href for href, _ in parser.links] + parser.image_sources:
        if not _local_target_exists(page, target):
            errors.append(f"missing local target: {target}")

    return errors


def check_figure3_page(page: Path) -> list[str]:
    """Check Figure 3 literals and source links after Sphinx rendering."""

    parser = RenderedDocumentParser()
    parser.feed(page.read_text(encoding="utf-8"))
    errors: list[str] = []
    missing_code = sorted(FIGURE3_EXPECTED_CODE.difference(parser.code_values))
    if missing_code:
        errors.append(f"missing Figure 3 code literals: {missing_code}")
    split_code = sorted(
        {
            value
            for value in parser.code_values
            if re.sub(r"\s+", "", value) in FIGURE3_EXPECTED_CODE
            and value != re.sub(r"\s+", "", value)
        }
    )
    if split_code:
        errors.append(f"split Figure 3 code literals: {split_code}")
    visible_text = " ".join(" ".join(parser.visible_parts).split())
    if re.search(r"\.py\s*<ht\s*tps?://", visible_text):
        errors.append("raw Figure 3 RST link markup")
    link_targets = {href for href, _ in parser.links}
    for expected in FIGURE3_SOURCE_URLS | FIGURE3_TAUREX_URLS:
        if expected not in link_targets:
            errors.append(f"missing Figure 3 source link: {expected}")
    for target in [href for href, _ in parser.links] + parser.image_sources:
        if not _local_target_exists(page, target):
            errors.append(f"missing local target: {target}")
    return errors


def main() -> None:
    build_root = Path(__file__).resolve().parents[1] / "_build" / "html"
    pages = (
        build_root
        / "ja"
        / "meltyq"
        / "meltyq_figure8_forward_comparison_ja.html",
        build_root
        / "en"
        / "meltyq"
        / "meltyq_figure8_forward_comparison_en.html",
    )
    figure3_pages = (
        build_root
        / "ja"
        / "meltyq"
        / "meltyq_figure3_forward_preparation_ja.html",
        build_root
        / "en"
        / "meltyq"
        / "meltyq_figure3_forward_preparation_en.html",
    )
    failures: list[str] = []
    for page in pages:
        if not page.is_file():
            failures.append(f"missing rendered page: {page}")
            continue
        failures.extend(f"{page.name}: {error}" for error in check_page(page))
    for page in figure3_pages:
        if not page.is_file():
            failures.append(f"missing rendered page: {page}")
            continue
        failures.extend(
            f"{page.name}: {error}" for error in check_figure3_page(page)
        )
    if failures:
        raise SystemExit("Rendered HTML check failed:\n" + "\n".join(failures))
    print("Rendered HTML checks passed for Japanese and English editions.")


if __name__ == "__main__":
    main()
