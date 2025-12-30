#!/usr/bin/env python3
"""
Generate camera-ready TikZ figure from experiment results.
Parses all CSV files and outputs a complete LaTeX figure.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("/scratch/local/arra4944_images/openbuildings/results")

# Sigma values in order
SIGMAS = [0, 1, 2, 3, 4, 5, 10, 20, 40]

def parse_results():
    """Parse all CSV files and organize by region/embedding."""
    results = defaultdict(lambda: defaultdict(dict))

    for csv_file in RESULTS_DIR.glob("*.csv"):
        # Parse filename: region_embedding_SH{L}_Slep{L}_seed{S}.csv
        name = csv_file.stem
        parts = name.split("_")

        # Find embedding type and L values
        region = parts[0]
        embedding = parts[1]  # alphaearth or galileo

        L_sh = None
        L_slep = None
        for p in parts:
            if p.startswith("SH"):
                L_sh = int(p[2:])
            elif p.startswith("Slep"):
                L_slep = int(p[4:])

        if L_sh is None or L_slep is None:
            continue

        df = pd.read_csv(csv_file)

        # Store results indexed by sigma
        key = f"SH{L_sh}_Slep{L_slep}"
        for _, row in df.iterrows():
            sigma = int(row["sigma_km"])

            # Image embedding R² (baseline)
            if embedding == "alphaearth":
                img_r2 = row["R2_ALPHAEARTH"]
            else:
                img_r2 = row["R2_GALILEO"]

            results[(region, embedding)][key][sigma] = {
                "img": img_r2,
                "sh": row["R2_SH"],
                "slep": row["R2_Slepian"],
                "L_sh": L_sh,
                "L_slep": L_slep,
            }

    return results


def format_coords(values, precision=4):
    """Format list of (sigma, value) as TikZ coordinates."""
    coords = []
    for sigma in SIGMAS:
        if sigma in values:
            coords.append(f"({sigma},{values[sigma]:.{precision}f})")
    return " ".join(coords)


def generate_macros(results):
    """Generate LaTeX macro definitions."""
    lines = []

    # Region name mapping
    region_names = {
        "southflorida": "SFla",
        "dhaka": "Dhaka",
        "maharashtra": "Maha",
        "mexicocity": "Mex"
    }

    emb_names = {"alphaearth": "AE", "galileo": "G"}

    for region in ["dhaka", "mexicocity", "maharashtra", "southflorida"]:
        for emb in ["alphaearth", "galileo"]:
            key = (region, emb)
            if key not in results:
                lines.append(f"% No data for {region} {emb}")
                continue

            data = results[key]
            rname = region_names[region]
            ename = emb_names[emb]

            # Image-only baseline (same across all experiments, take from first)
            first_exp = list(data.values())[0]
            img_vals = {s: d["img"] for s, d in first_exp.items()}
            lines.append(f"\\def\\{rname}Img{ename}{{{format_coords(img_vals)}}}")

            # SH L=10 (take from any SH10_* experiment)
            sh10_exp = None
            sh40_exp = None
            for exp_key, exp_data in data.items():
                if "SH10" in exp_key:
                    sh10_exp = exp_data
                if "SH40" in exp_key:
                    sh40_exp = exp_data

            if sh10_exp:
                sh10_vals = {s: d["sh"] for s, d in sh10_exp.items()}
                lines.append(f"\\def\\{rname}SHten{ename}{{{format_coords(sh10_vals)}}}")

            if sh40_exp:
                sh40_vals = {s: d["sh"] for s, d in sh40_exp.items()}
                lines.append(f"\\def\\{rname}SHforty{ename}{{{format_coords(sh40_vals)}}}")

            # Slepian L=40, 64, 96 (take from SH40_* experiments for best SH comparison)
            for L_slep, lname in [(40, "forty"), (64, "sixtyfour"), (96, "ninetysix")]:
                exp_key = f"SH40_Slep{L_slep}"
                if exp_key in data:
                    slep_vals = {s: d["slep"] for s, d in data[exp_key].items()}
                    lines.append(f"\\def\\{rname}SL{lname}{ename}{{{format_coords(slep_vals)}}}")

            lines.append("")

    return "\n".join(lines)


def generate_figure(results):
    """Generate complete LaTeX figure."""

    macros = generate_macros(results)

    # Calculate ymin for each region based on actual data
    region_ymins = {}
    for region in ["dhaka", "mexicocity", "maharashtra", "southflorida"]:
        min_val = 1.0
        for emb in ["alphaearth", "galileo"]:
            key = (region, emb)
            if key in results:
                for exp_data in results[key].values():
                    for sigma_data in exp_data.values():
                        min_val = min(min_val, sigma_data["img"], sigma_data["sh"], sigma_data["slep"])
        # Round down to nearest 0.05 and subtract 0.02 for padding
        region_ymins[region] = max(0.30, round(min_val * 20) / 20 - 0.03)

    figure = f'''% =============================================================================
% Camera-ready figure: Slepian vs SH positional encodings
% Auto-generated from experiment results
% =============================================================================

% Preamble (include in document):
% \\usepackage{{pgfplots}}
% \\usepackage{{caption}}
% \\pgfplotsset{{compat=1.18}}
% \\usepgfplotslibrary{{groupplots,fillbetween}}

% Colors
\\definecolor{{slepianpurple1}}{{RGB}}{{185,121,180}}
\\definecolor{{slepianpurple2}}{{RGB}}{{160,76,150}}
\\definecolor{{slepianpurple3}}{{RGB}}{{135,31,120}}
\\definecolor{{SHorange}}{{RGB}}{{225,182,93}}
\\definecolor{{SHblue}}{{RGB}}{{63,100,147}}
\\definecolor{{imggray}}{{RGB}}{{120,120,120}}

% ----------------------------
% Data macros
% ----------------------------
{macros}

% ----------------------------
% Plot styles
% ----------------------------
\\pgfplotsset{{
  imgline/.style={{imggray, dashed, line width=0.9pt}},
  sh10/.style={{SHorange, mark=o, mark size=1pt, line width=0.9pt}},
  sh40/.style={{SHblue, mark=square, mark size=1pt, line width=0.95pt}},
  sl40/.style={{slepianpurple1, mark=*, mark size=1pt, line width=0.9pt}},
  sl64/.style={{slepianpurple2, mark=triangle, mark size=1.1pt, line width=0.9pt}},
  sl96/.style={{slepianpurple3, mark=diamond, mark size=1.1pt, line width=0.95pt}},
  gainSH/.style={{draw=none, fill=SHblue, fill opacity=0.08}},
  gainSL/.style={{draw=none, fill=slepianpurple3, fill opacity=0.12}},
}}

\\begin{{figure*}}[t!]
\\centering

\\begin{{minipage}}[t]{{0.78\\linewidth}}
\\centering
\\begin{{tikzpicture}}

\\begin{{groupplot}}[
  group style={{
    group size=2 by 4,
    horizontal sep=0.65cm,
    vertical sep=0.55cm,
    xlabels at=edge bottom,
    ylabels at=edge left,
  }},
  width=0.48\\linewidth,
  height=0.22\\linewidth,
  xmin=-2, xmax=42,
  xtick={{0,5,10,20,40}},
  grid=major,
  grid style={{dashed, gray!20}},
  xlabel={{$\\sigma$ [km]}},
  ylabel={{Test R$^2$}},
  tick label style={{font=\\scriptsize}},
  label style={{font=\\scriptsize}},
  title style={{font=\\footnotesize\\bfseries, yshift=-1pt}},
  legend to name=globlegend,
  legend style={{draw=none, fill=none, font=\\scriptsize, cells={{anchor=west}}}},
  legend columns=3,
  clip=true,
]

% =========================
% Row 1: Dhaka
% =========================
\\nextgroupplot[title={{Dhaka (AlphaEarth)}}, ymin={region_ymins["dhaka"]:.2f}, ymax=1.02]
  \\addplot[imgline, name path=img] coordinates {{\\DhakaImgAE}};      \\addlegendentry{{Image-only}}
  \\addplot[sh10]  coordinates {{\\DhakaSHtenAE}};                     \\addlegendentry{{SH $L{{=}}10$}}
  \\addplot[sh40, name path=sh] coordinates {{\\DhakaSHfortyAE}};      \\addlegendentry{{SH $L{{=}}40$}}
  \\addplot[sl40]  coordinates {{\\DhakaSLfortyAE}};                   \\addlegendentry{{Slep. $L{{=}}40$}}
  \\addplot[sl64]  coordinates {{\\DhakaSLsixtyfourAE}};               \\addlegendentry{{Slep. $L{{=}}64$}}
  \\addplot[sl96, name path=sl] coordinates {{\\DhakaSLninetysixAE}};  \\addlegendentry{{Slep. $L{{=}}96$}}
  \\addplot[gainSH] fill between[of=img and sh];
  \\addplot[gainSL] fill between[of=sh and sl];

\\nextgroupplot[title={{Dhaka (Galileo)}}, ymin={region_ymins["dhaka"]:.2f}, ymax=1.02]
  \\addplot[imgline, name path=img] coordinates {{\\DhakaImgG}};
  \\addplot[sh10]  coordinates {{\\DhakaSHtenG}};
  \\addplot[sh40, name path=sh] coordinates {{\\DhakaSHfortyG}};
  \\addplot[sl40]  coordinates {{\\DhakaSLfortyG}};
  \\addplot[sl64]  coordinates {{\\DhakaSLsixtyfourG}};
  \\addplot[sl96, name path=sl] coordinates {{\\DhakaSLninetysixG}};
  \\addplot[gainSH] fill between[of=img and sh];
  \\addplot[gainSL] fill between[of=sh and sl];

% =========================
% Row 2: Mexico City
% =========================
\\nextgroupplot[title={{Mexico City (AlphaEarth)}}, ymin={region_ymins["mexicocity"]:.2f}, ymax=1.02]
  \\addplot[imgline, name path=img] coordinates {{\\MexImgAE}};
  \\addplot[sh10]  coordinates {{\\MexSHtenAE}};
  \\addplot[sh40, name path=sh] coordinates {{\\MexSHfortyAE}};
  \\addplot[sl40]  coordinates {{\\MexSLfortyAE}};
  \\addplot[sl64]  coordinates {{\\MexSLsixtyfourAE}};
  \\addplot[sl96, name path=sl] coordinates {{\\MexSLninetysixAE}};
  \\addplot[gainSH] fill between[of=img and sh];
  \\addplot[gainSL] fill between[of=sh and sl];

\\nextgroupplot[title={{Mexico City (Galileo)}}, ymin={region_ymins["mexicocity"]:.2f}, ymax=1.02]
  \\addplot[imgline, name path=img] coordinates {{\\MexImgG}};
  \\addplot[sh10]  coordinates {{\\MexSHtenG}};
  \\addplot[sh40, name path=sh] coordinates {{\\MexSHfortyG}};
  \\addplot[sl40]  coordinates {{\\MexSLfortyG}};
  \\addplot[sl64]  coordinates {{\\MexSLsixtyfourG}};
  \\addplot[sl96, name path=sl] coordinates {{\\MexSLninetysixG}};
  \\addplot[gainSH] fill between[of=img and sh];
  \\addplot[gainSL] fill between[of=sh and sl];

% =========================
% Row 3: Maharashtra
% =========================
\\nextgroupplot[title={{Maharashtra (AlphaEarth)}}, ymin={region_ymins["maharashtra"]:.2f}, ymax=1.02]
  \\addplot[imgline, name path=img] coordinates {{\\MahaImgAE}};
  \\addplot[sh10]  coordinates {{\\MahaSHtenAE}};
  \\addplot[sh40, name path=sh] coordinates {{\\MahaSHfortyAE}};
  \\addplot[sl40]  coordinates {{\\MahaSLfortyAE}};
  \\addplot[sl64]  coordinates {{\\MahaSLsixtyfourAE}};
  \\addplot[sl96, name path=sl] coordinates {{\\MahaSLninetysixAE}};
  \\addplot[gainSH] fill between[of=img and sh];
  \\addplot[gainSL] fill between[of=sh and sl];

\\nextgroupplot[title={{Maharashtra (Galileo)}}, ymin={region_ymins["maharashtra"]:.2f}, ymax=1.02]
  \\addplot[imgline, name path=img] coordinates {{\\MahaImgG}};
  \\addplot[sh10]  coordinates {{\\MahaSHtenG}};
  \\addplot[sh40, name path=sh] coordinates {{\\MahaSHfortyG}};
  \\addplot[sl40]  coordinates {{\\MahaSLfortyG}};
  \\addplot[sl64]  coordinates {{\\MahaSLsixtyfourG}};
  \\addplot[sl96, name path=sl] coordinates {{\\MahaSLninetysixG}};
  \\addplot[gainSH] fill between[of=img and sh];
  \\addplot[gainSL] fill between[of=sh and sl];

% =========================
% Row 4: South Florida
% =========================
\\nextgroupplot[title={{South Florida (AlphaEarth)}}, ymin={region_ymins["southflorida"]:.2f}, ymax=1.02]
  \\addplot[imgline, name path=img] coordinates {{\\SFlaImgAE}};
  \\addplot[sh10]  coordinates {{\\SFlaSHtenAE}};
  \\addplot[sh40, name path=sh] coordinates {{\\SFlaSHfortyAE}};
  \\addplot[sl40]  coordinates {{\\SFlaSLfortyAE}};
  \\addplot[sl64]  coordinates {{\\SFlaSLsixtyfourAE}};
  \\addplot[sl96, name path=sl] coordinates {{\\SFlaSLninetysixAE}};
  \\addplot[gainSH] fill between[of=img and sh];
  \\addplot[gainSL] fill between[of=sh and sl];

\\nextgroupplot[title={{South Florida (Galileo)}}, ymin={region_ymins["southflorida"]:.2f}, ymax=1.02]
  \\addplot[imgline, name path=img] coordinates {{\\SFlaImgG}};
  \\addplot[sh10]  coordinates {{\\SFlaSHtenG}};
  \\addplot[sh40, name path=sh] coordinates {{\\SFlaSHfortyG}};
  \\addplot[sl40]  coordinates {{\\SFlaSLfortyG}};
  \\addplot[sl64]  coordinates {{\\SFlaSLsixtyfourG}};
  \\addplot[sl96, name path=sl] coordinates {{\\SFlaSLninetysixG}};
  \\addplot[gainSH] fill between[of=img and sh];
  \\addplot[gainSL] fill between[of=sh and sl];

\\end{{groupplot}}

% Legend centered above plots
\\node[anchor=south] at
  ($(group c1r1.north)!0.5!(group c2r1.north) + (0,0.3cm)$)
  {{\\pgfplotslegendfromname{{globlegend}}}};

\\end{{tikzpicture}}
\\end{{minipage}}
\\hfill
\\begin{{minipage}}[t]{{0.20\\linewidth}}
\\vspace{{0pt}}
\\captionof{{figure}}{{\\textbf{{Building density regression with Slepian positional encodings.}}
Each row shows a different geographic region; columns compare AlphaEarth (63-dim) and Galileo (128-dim) image embeddings.
\\textit{{Gray dashed}}: image embedding only.
\\textit{{Orange/blue}}: spherical harmonics ($L{{=}}10, 40$).
\\textit{{Purple}}: Slepian functions ($L{{=}}40, 64, 96$).
Slepian encodings consistently outperform global SH at intermediate scales ($\\sigma{{=}}2$--$10$\\,km), with gains of up to 15\\% R$^2$.
Shaded regions show improvements: SH over image-only (blue) and Slepian over SH (purple).}}
\\label{{fig:slepian_vs_sh}}
\\end{{minipage}}

\\end{{figure*}}
'''

    return figure


def main():
    print("Parsing experiment results...")
    results = parse_results()

    print(f"Found results for {len(results)} region/embedding combinations:")
    for key in sorted(results.keys()):
        print(f"  {key[0]}/{key[1]}: {len(results[key])} experiments")

    print("\nGenerating figure...")
    figure = generate_figure(results)

    output_path = Path("/projects/arra4944/SlepianPosEnc/src/slepian_image_augmentation/figure_camera_ready.tex")
    output_path.write_text(figure)
    print(f"\nSaved to: {output_path}")

    # Also print the macros section for inspection
    print("\n" + "="*60)
    print("DATA MACROS (for verification):")
    print("="*60)
    print(generate_macros(results))


if __name__ == "__main__":
    main()
