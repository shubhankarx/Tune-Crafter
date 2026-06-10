// AmlClient — the single place that talks to the AML Docker server.
//
// Centralizes the localhost:5001 URLs (previously duplicated in GestureComponent),
// adds a reachability probe for the connection badge, and returns TYPED results so
// the UI can tell "server is down" apart from "training failed".
//
// The server (discreteDocker2/server.py) exposes only `POST /train/<nIter>` and
// must be running for training to work — see docs/aml-server-setup.md.

import { AmlPayload } from "./AmlModel";

export const AML_BASE_URL = "http://localhost:5001";
const AML_TRAIN_URL = `${AML_BASE_URL}/train/`;

export type AmlTrainResult =
    | { ok: true; payload: AmlPayload }
    | { ok: false; kind: "unreachable" | "http" | "bad-shape"; message: string };

// Is the server reachable? We don't have a health route, so we just check that
// SOMETHING answers on :5001 — any HTTP response (even a 404) proves reachability;
// only a network error / timeout means "unreachable". `no-cors` keeps this immune
// to CORS preflight quirks since we never read the response body here.
export async function pingAmlServer(timeoutMs = 2500): Promise<boolean> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        await fetch(`${AML_BASE_URL}/`, {
            method: "GET",
            mode: "no-cors",
            cache: "no-cache",
            signal: controller.signal,
        });
        return true;
    } catch {
        return false;
    } finally {
        clearTimeout(timer);
    }
}

// Train an AML model. Returns the raw `{ edges, classes }` payload on success, or a
// tagged error the panel can render distinctly. Training runs the AML engine for
// `nIter` batches and can take a while, hence the generous timeout.
export async function trainAml(
    x: number[][],
    y: string[],
    nIter: number,
    timeoutMs = 120000,
): Promise<AmlTrainResult> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        const response = await fetch(`${AML_TRAIN_URL}${nIter}`, {
            method: "POST",
            mode: "cors",
            cache: "no-cache",
            credentials: "omit",
            headers: { "Content-Type": "application/json" },
            redirect: "follow",
            referrerPolicy: "no-referrer",
            body: JSON.stringify({ x, y }),
            signal: controller.signal,
        });

        if (!response.ok) {
            return { ok: false, kind: "http", message: `Server returned HTTP ${response.status}.` };
        }

        const payload = (await response.json()) as AmlPayload;
        if (!payload || typeof payload !== "object" || !payload.classes || !payload.edges) {
            return { ok: false, kind: "bad-shape", message: "Response was missing edges/classes." };
        }
        return { ok: true, payload };
    } catch (error) {
        const message = error instanceof Error ? error.message : "Network error.";
        return { ok: false, kind: "unreachable", message };
    } finally {
        clearTimeout(timer);
    }
}
