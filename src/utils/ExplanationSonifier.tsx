// ExplanationSonifier — "the explanation IS the sound".
//
// Core artifact for the "explainability as music" direction: it turns a model's
// INTERNAL interpretable state into an audible chord, so a user can HEAR why the
// model decided as it did.
//
//   - Each class atom maps to a note in a chord.
//   - A PRESENT atom plays its note; a MISSING atom is muted → the chord goes
//     thin/hollow exactly in proportion to how many atoms are missing.
//   - Confidence drives timbre (a low-pass cutoff): confident = bright, unsure = dull.
//
// It is deliberately decoupled from any model. It consumes a plain `Explanation`
// object, so today we can feed it a STUB derived from the local TF.js model, and
// later feed it the real AML atom/miss structure from the AML server with zero
// changes here. (Direction 1 in docs/research-notes.md.)

export interface Explanation {
    predictedClass: string;
    confidence: number;   // 0..1  (AML: 1 - P(>=misses | negatives))
    atomsTotal: number;   // number of atoms the model has for the predicted class
    atomsMissing: number; // how many of those atoms are NOT matched by this input
}

// A simple, pleasant chord (semitone offsets from a root) we mute notes out of.
// Major 9th — enough notes to make "missing atoms" audibly thin out the harmony.
const CHORD_SEMITONES = [0, 4, 7, 11, 14, 17, 21, 24];
const ROOT_HZ = 220; // A3

const semitoneToHz = (semitone: number): number => ROOT_HZ * Math.pow(2, semitone / 12);

export class ExplanationSonifier {
    private audioContext: AudioContext | null = null;

    private ensureContext(): AudioContext | null {
        if (!this.audioContext) {
            try {
                this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
            } catch (error) {
                console.error('ExplanationSonifier: AudioContext init failed', error);
                return null;
            }
        }
        return this.audioContext;
    }

    // Play the explanation as a chord. `atomsPresent / atomsTotal` of the chord's
    // notes sound; the rest are muted. Confidence sets the brightness (filter cutoff).
    play(explanation: Explanation, durationSec = 1.2): void {
        const ctx = this.ensureContext();
        if (!ctx) return;

        const { confidence, atomsTotal, atomsMissing } = explanation;
        const present = Math.max(0, atomsTotal - atomsMissing);
        // Fraction of the chord that should sound (0..1).
        const presentFraction = atomsTotal > 0 ? present / atomsTotal : 0;
        const notesToPlay = Math.round(presentFraction * CHORD_SEMITONES.length);

        const now = ctx.currentTime;

        // Confidence → brightness: unsure predictions sound dull/filtered.
        const filter = ctx.createBiquadFilter();
        const master = ctx.createGain();
        master.gain.setValueAtTime(0.0001, now);
        master.gain.exponentialRampToValueAtTime(0.25, now + 0.05);
        master.gain.exponentialRampToValueAtTime(0.0001, now + durationSec);

        filter.type = 'lowpass';
        // 600 Hz (muffled, unsure) → 6000 Hz (bright, confident).
        filter.frequency.setValueAtTime(600 + 5400 * Math.min(1, Math.max(0, confidence)), now);
        master.connect(filter);
        filter.connect(ctx.destination);

        for (let i = 0; i < CHORD_SEMITONES.length; i++) {
            const osc = ctx.createOscillator();
            const noteGain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(semitoneToHz(CHORD_SEMITONES[i]), now);
            // Present atoms sound; missing atoms are muted (gain 0).
            const audible = i < notesToPlay;
            noteGain.gain.setValueAtTime(audible ? 1 / CHORD_SEMITONES.length : 0, now);
            osc.connect(noteGain);
            noteGain.connect(master);
            osc.start(now);
            osc.stop(now + durationSec);
        }
    }

    // A short text gloss to pair with the sound (for the explanation panel / logging).
    describe(explanation: Explanation): string {
        const { predictedClass, confidence, atomsTotal, atomsMissing } = explanation;
        const present = Math.max(0, atomsTotal - atomsMissing);
        const pct = Math.round(confidence * 100);
        if (atomsMissing === 0) {
            return `${predictedClass}: all ${atomsTotal} atoms matched — full chord, ${pct}% sure.`;
        }
        return `${predictedClass}: ${present}/${atomsTotal} atoms matched (${atomsMissing} missing → muted notes), ${pct}% sure.`;
    }
}

// --- STUB explanation generator -------------------------------------------------
// Until the AML server returns real atoms/misses, derive a plausible Explanation
// from any classifier's confidence so the sonification is demoable TODAY.
// Replace this with the AML server's atom/miss JSON when available.
export const stubExplanationFromConfidence = (
    predictedClass: string,
    confidence: number,
    atomsTotal = 8,
): Explanation => {
    const clamped = Math.min(1, Math.max(0, confidence));
    // Lower confidence ⇒ more "missing" atoms ⇒ thinner chord.
    const atomsMissing = Math.round((1 - clamped) * atomsTotal);
    return { predictedClass, confidence: clamped, atomsTotal, atomsMissing };
};
