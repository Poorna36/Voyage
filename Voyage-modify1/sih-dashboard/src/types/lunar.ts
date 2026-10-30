export type SensorType = 'OHRC' | 'TMC-2' | 'IIRS';

export type ReferenceDatasetType = 'LRO' | 'Kaguya';

export interface SensorSpec {
  id: SensorType;
  name: string;
  fullName: string;
  resolution: string;
  resolutionMeters: number;
  spectralBands: string;
  swathWidth: string;
  spectralRange: string;
  primaryRole: string;
  description: string;
  pdsSchema: string;
  sampleImg: string;
  color: string;
}

export interface PDS4Metadata {
  productId: string;
  targetName: string;
  missionPhase: string;
  instrumentId: SensorType;
  instrumentName: string;
  acquisitionDate: string;
  acquisitionTime: string;
  imageDimensions: {
    samples: number;
    lines: number;
  };
  resolution: {
    groundSampleDistance: number;
    unit: string;
  };
  geometry: {
    subSpacecraftLatitude: number;
    subSpacecraftLongitude: number;
    spacecraftAltitudeKm: number;
    centerLatitude: number;
    centerLongitude: number;
    incidenceAngleDeg: number;
    emissionAngleDeg: number;
    phaseAngleDeg: number;
    solarAzimuthDeg: number;
    solarElevationDeg: number;
    slantDistanceKm: number;
  };
  footprint: {
    cornerCoordinates: [number, number][]; // [Lat, Lon]
    areaKm2: number;
    boundingPolygon: string;
  };
  radiometry: {
    exposureDurationMs: number;
    calibrationVersion: string;
    gainSetting: string;
    unmodifiedRadiometry: boolean;
  };
  pdsXmlContent: string;
}

export interface LunarDatasetPreset {
  id: string;
  name: string;
  region: string;
  description: string;
  targetFeature: string;
  centerCoordinates: {
    lat: number;
    lon: number;
  };
  ohrc: PDS4Metadata;
  tmc2: PDS4Metadata;
  iirs: PDS4Metadata;
  lroReference: {
    mission: string;
    instrument: string;
    resolution: string;
    sunAngle: number;
    centerCoordinates: { lat: number; lon: number };
    image: string;
  };
  images: {
    ohrc: string;
    tmc2: string;
    iirs: string;
  };
  overlapStats: {
    commonAreaKm2: number;
    overlapPercentageOHRC: number;
    overlapPercentageTMC2: number;
    overlapPercentageIIRS: number;
    illuminationDiscrepancyDeg: number;
    scaleDifferenceRatio: string;
  };
}
