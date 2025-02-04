import * as Tone from 'tone';
import { mapColorToInstrument } from './soundMappings';

const instrumentCache = {};

const getInstrument = (instrumentName) => {
  if (!instrumentCache[instrumentName]) {
    switch (instrumentName) {
      case 'Synth':
        instrumentCache[instrumentName] = new Tone.Synth().toDestination();
        break;
      case 'AMSynth':
        instrumentCache[instrumentName] = new Tone.AMSynth().toDestination();
        break;
      case 'DuoSynth':
        instrumentCache[instrumentName] = new Tone.DuoSynth().toDestination();
        break;
      case 'FMSynth':
        instrumentCache[instrumentName] = new Tone.FMSynth().toDestination();
        break;
      case 'MembraneSynth':
        instrumentCache[instrumentName] = new Tone.MembraneSynth().toDestination();
        break;
      case 'MetalSynth':
        instrumentCache[instrumentName] = new Tone.MetalSynth().toDestination();
        break;
      case 'MonoSynth':
        instrumentCache[instrumentName] = new Tone.MonoSynth().toDestination();
        break;
      case 'NoiseSynth':
        instrumentCache[instrumentName] = new Tone.NoiseSynth().toDestination();
        break;
      case 'PluckSynth':
        instrumentCache[instrumentName] = new Tone.PluckSynth().toDestination();
        break;
      case 'PolySynth':
        instrumentCache[instrumentName] = new Tone.PolySynth().toDestination();
        break;
      case 'Sampler':
        instrumentCache[instrumentName] = new Tone.Sampler({
          urls: {
            A1: 'A1.mp3',
            A2: 'A2.mp3',
          },
          baseUrl: 'https://tonejs.github.io/audio/salamander/'
        }).toDestination();
        break;
      default:
        instrumentCache[instrumentName] = new Tone.Synth().toDestination();
        break;
    }
  }
  return instrumentCache[instrumentName];
};

export const playSound = (color, note) => {
  const instrumentName = mapColorToInstrument[color];
  const instrument = getInstrument(instrumentName);
  instrument.triggerAttackRelease(note, '8n');
};
