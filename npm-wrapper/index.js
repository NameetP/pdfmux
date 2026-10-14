#!/usr/bin/env node
/**
 * pdfmux-mcp — npm wrapper for the pdfmux MCP server.
 *
 * Usage:
 *   npx -y pdfmux-mcp          # runs pdfmux serve (stdio)
 *   npx -y pdfmux-mcp --http   # runs pdfmux serve --http
 *
 * This wrapper ensures Python + pdfmux are installed, then delegates
 * to `pdfmux serve`. If pdfmux isn't installed, it installs it via pip.
 *
 * Claude Desktop / Cursor config:
 *   { "mcpServers": { "pdfmux": { "command": "npx", "args": ["-y", "pdfmux-mcp"] } } }
 */

const { execSync, spawn } = require("child_process");
const { existsSync } = require("fs");

const PDFMUX_MIN_VERSION = "1.4.0";

function findPython() {
  for (const cmd of ["python3", "python"]) {
    try {
      const version = execSync(`${cmd} --version 2>&1`, { encoding: "utf8" });
      const match = version.match(/Python (\d+)\.(\d+)/);
      if (match && (parseInt(match[1]) > 3 || (parseInt(match[1]) === 3 && parseInt(match[2]) >= 11))) {
        return cmd;
      }
    } catch {}
  }
  return null;
}

function getInstalledVersion(python) {
  try {
    return execSync(`${python} -c "import pdfmux; print(pdfmux.__version__)"`, {
      encoding: "utf8",
      stdio: ["pipe", "pipe", "pipe"],
    }).trim();
  } catch {
    return null;
  }
}

function versionAtLeast(version, minVersion) {
  const parts = (v) => v.split(".").map((n) => parseInt(n, 10) || 0);
  const a = parts(version);
  const b = parts(minVersion);
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const x = a[i] || 0;
    const y = b[i] || 0;
    if (x !== y) return x > y;
  }
  return true;
}

function installPdfmux(python) {
  console.error("[pdfmux-mcp] Installing pdfmux...");
  try {
    execSync(`${python} -m pip install "pdfmux[serve]>=${PDFMUX_MIN_VERSION}"`, {
      stdio: ["pipe", "inherit", "inherit"],
    });
    console.error("[pdfmux-mcp] pdfmux installed successfully.");
    return true;
  } catch (e) {
    console.error("[pdfmux-mcp] Failed to install pdfmux:", e.message);
    return false;
  }
}

function main() {
  const python = findPython();
  if (!python) {
    console.error(
      "[pdfmux-mcp] Python 3.11+ not found. Install from https://python.org"
    );
    process.exit(1);
  }

  // Bug (reported as "pdfmux-mcp garbles Korean text, pdfmux CLI doesn't" —
  // NameetP/pdfmux#10): this check used to only ask "is pdfmux importable at
  // all," so a stale pdfmux already present on whatever `python3`/`python`
  // this wrapper resolves to (which can be a different install than the
  // user's own `pdfmux` CLI binary) was accepted silently, even below
  // PDFMUX_MIN_VERSION. Two installs drifting apart is exactly the kind of
  // silent inconsistency this product's extraction claims are supposed to
  // rule out — it must not exist in its own install path.
  const installedVersion = getInstalledVersion(python);
  if (!installedVersion || !versionAtLeast(installedVersion, PDFMUX_MIN_VERSION)) {
    if (installedVersion) {
      console.error(
        `[pdfmux-mcp] Installed pdfmux ${installedVersion} is older than the required ${PDFMUX_MIN_VERSION} — upgrading.`
      );
    }
    if (!installPdfmux(python)) {
      process.exit(1);
    }
  }

  // Forward all args to pdfmux serve
  const args = process.argv.slice(2);
  const serveArgs = ["-m", "pdfmux.cli", "serve", ...args];

  const child = spawn(python, serveArgs, {
    stdio: "inherit",
    env: { ...process.env },
  });

  child.on("exit", (code) => process.exit(code || 0));
  child.on("error", (err) => {
    console.error("[pdfmux-mcp] Failed to start:", err.message);
    process.exit(1);
  });

  // Forward signals
  for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) {
    process.on(sig, () => child.kill(sig));
  }
}

main();
