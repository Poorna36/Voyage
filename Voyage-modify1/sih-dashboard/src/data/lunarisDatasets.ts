import type { SensorSpec, LunarDatasetPreset } from '../types/lunar';

import ohrcImg from '../assets/images/ohrc_lunar_crater_1788336805774.jpg';
import tmc2Img from '../assets/images/tmc2_terrain_context_1788336820221.jpg';
import iirsImg from '../assets/images/iirs_hyperspectral_overlay_1788336834453.jpg';
import heroImg from '../assets/images/lunar_surface_hero_1788336791925.jpg';
import lroImg from '../assets/images/lro_reference_baseline_1788336850293.jpg';
import moon8kImg from '../assets/images/moon_8k.jpg';
import chandrayaanOrbiterImg from '../assets/images/chandrayaan2_orbiter.jpg';
import chandrayaan8kImg from '../assets/images/chandrayaan2_8k.jpg';
import chandrayaanTransparentImg from '../assets/images/chandrayaan2_8k_transparent.png';

export { ohrcImg, tmc2Img, iirsImg, heroImg, lroImg, moon8kImg, chandrayaanOrbiterImg, chandrayaan8kImg, chandrayaanTransparentImg };

export const SENSOR_SPECS: Record<string, SensorSpec> = {
  OHRC: {
    id: 'OHRC',
    name: 'OHRC',
    fullName: 'Orbiter High Resolution Camera',
    resolution: '0.25 m / px',
    resolutionMeters: 0.25,
    spectralBands: '1 Panchromatic (450 – 900 nm)',
    swathWidth: '3.0 km at 100 km altitude',
    spectralRange: '450 – 900 nm (High SNR CCD)',
    primaryRole: 'High-resolution optical imagery for detailed lunar surface observation & hazard avoidance',
    description: 'Extremely high-resolution imagery covering precise lunar patches. Capable of resolving boulders down to 25 cm across for landing site characterization and micro-crater morphometry.',
    pdsSchema: 'urn:isro:ch2:ohrc:pds4:schema:v1.2',
    sampleImg: ohrcImg,
    color: '#D6C38B',
  },
  'TMC-2': {
    id: 'TMC-2',
    name: 'TMC-2',
    fullName: 'Terrain Mapping Camera-2',
    resolution: '5.0 m / px',
    resolutionMeters: 5.0,
    spectralBands: '3 Views: Fore, Nadir, Aft (500 – 850 nm)',
    swathWidth: '20.0 km stereo swath',
    spectralRange: '500 – 850 nm (Triplet CCD Optics)',
    primaryRole: 'Stereo terrain imagery providing wider geological context & Digital Elevation Models (DEM)',
    description: 'Provides medium-resolution imagery and stereoscopic surface geometry across three optical look-angles (Fore +26°, Nadir 0°, Aft -26°) to generate sub-meter accurate elevation contours.',
    pdsSchema: 'urn:isro:ch2:tmc2:pds4:schema:v2.0',
    sampleImg: tmc2Img,
    color: '#E7E3D9',
  },
  IIRS: {
    id: 'IIRS',
    name: 'IIRS',
    fullName: 'Imaging Infrared Spectrometer',
    resolution: '80.0 m / px',
    resolutionMeters: 80.0,
    spectralBands: '256 Contiguous Spectral Bands',
    swathWidth: '20.0 km hyperspectral swath',
    spectralRange: '0.8 – 5.0 µm (NIR to Thermal/SWIR)',
    primaryRole: 'Hyperspectral imagery revealing mineralogical composition & hydroxyl/water signatures beyond visible light',
    description: 'Captures approximately 256 spectral channels between 0.8 and 5.0 micrometers, discerning pyroxene, plagioclase, olivine, and surface hydration features independently of visible albedo.',
    pdsSchema: 'urn:isro:ch2:iirs:pds4:schema:v1.0',
    sampleImg: iirsImg,
    color: '#B7B5AE',
  },
};

export const LUNAR_PRESETS: LunarDatasetPreset[] = [
  {
    id: 'boguslawsky-site',
    name: 'Boguslawsky Crater Basin',
    region: 'South Polar Highlands (72.9° S, 53.2° E)',
    description: 'High-priority polar geological contact zone featuring steep shadow boundaries, multiple crater degradation stages, and complex solar incidence angles.',
    targetFeature: 'Boguslawsky-E Floor Regolith',
    centerCoordinates: {
      lat: -72.914,
      lon: 53.284,
    },
    images: {
      ohrc: ohrcImg,
      tmc2: tmc2Img,
      iirs: iirsImg,
    },
    lroReference: {
      mission: 'NASA Lunar Reconnaissance Orbiter (LRO)',
      instrument: 'Narrow Angle Camera (LROC-NAC)',
      resolution: '0.50 m / px',
      sunAngle: 78.4,
      centerCoordinates: { lat: -72.915, lon: 53.280 },
      image: lroImg,
    },
    overlapStats: {
      commonAreaKm2: 8.42,
      overlapPercentageOHRC: 98.4,
      overlapPercentageTMC2: 42.1,
      overlapPercentageIIRS: 21.0,
      illuminationDiscrepancyDeg: 34.6,
      scaleDifferenceRatio: '1 : 20 : 320 (OHRC vs TMC-2 vs IIRS)',
    },
    ohrc: {
      productId: 'CH2_OHR_NC_20230817T041219832_L1B',
      targetName: 'Moon',
      missionPhase: 'Extended Science Orbit',
      instrumentId: 'OHRC',
      instrumentName: 'Orbiter High Resolution Camera',
      acquisitionDate: '2023-08-17',
      acquisitionTime: '04:12:19.832 UTC',
      imageDimensions: { samples: 12000, lines: 9000 },
      resolution: { groundSampleDistance: 0.25, unit: 'm/pixel' },
      geometry: {
        subSpacecraftLatitude: -72.914,
        subSpacecraftLongitude: 53.284,
        spacecraftAltitudeKm: 99.4,
        centerLatitude: -72.9142,
        centerLongitude: 53.2841,
        incidenceAngleDeg: 42.8,
        emissionAngleDeg: 1.2,
        phaseAngleDeg: 43.1,
        solarAzimuthDeg: 124.6,
        solarElevationDeg: 47.2,
        slantDistanceKm: 99.42,
      },
      footprint: {
        cornerCoordinates: [
          [-72.880, 53.220],
          [-72.880, 53.348],
          [-72.948, 53.348],
          [-72.948, 53.220],
        ],
        areaKm2: 8.56,
        boundingPolygon: 'POLYGON((53.22 -72.88, 53.348 -72.88, 53.348 -72.948, 53.22 -72.948, 53.22 -72.88))',
      },
      radiometry: {
        exposureDurationMs: 4.8,
        calibrationVersion: 'ISRO-SAC-CAL-v3.1',
        gainSetting: 'HIGH_GAIN_CH01',
        unmodifiedRadiometry: true,
      },
      pdsXmlContent: `<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1" xmlns:isro="http://isro.gov.in/pds4/isro/v1">
  <Identification_Area>
    <logical_identifier>urn:isro:ch2:ohrc:data:ch2_ohr_nc_20230817t041219832_l1b</logical_identifier>
    <version_id>1.0</version_id>
    <title>Chandrayaan-2 OHRC Level-1B Calibrated Observation</title>
    <information_model_version>1.14.0.0</information_model_version>
    <product_class>Product_Observational</product_class>
  </Identification_Area>
  <Observation_Area>
    <Time_Coordinates>
      <start_date_time>2023-08-17T04:12:19.832Z</start_date_time>
      <stop_date_time>2023-08-17T04:12:35.120Z</stop_date_time>
    </Time_Coordinates>
    <Primary_Result_Summary>
      <purpose>Science</purpose>
      <processing_level>Calibrated</processing_level>
    </Primary_Result_Summary>
    <Investigation_Area>
      <name>CHANDRAYAAN-2</name>
      <type>Mission</type>
    </Investigation_Area>
    <Observing_System>
      <Observing_System_Component>
        <name>Orbiter High Resolution Camera</name>
        <type>Instrument</type>
      </Observing_System_Component>
    </Observing_System>
    <Target_Identification>
      <name>MOON</name>
      <type>Satellite</type>
    </Target_Identification>
    <isro:Geometry_Parameters>
      <isro:ground_sample_distance unit="m">0.25</isro:ground_sample_distance>
      <isro:center_latitude unit="deg">-72.9142</isro:center_latitude>
      <isro:center_longitude unit="deg">53.2841</isro:center_longitude>
      <isro:solar_incidence_angle unit="deg">42.8</isro:solar_incidence_angle>
      <isro:emission_angle unit="deg">1.2</isro:emission_angle>
      <isro:phase_angle unit="deg">43.1</isro:phase_angle>
      <isro:spacecraft_altitude unit="km">99.40</isro:spacecraft_altitude>
    </isro:Geometry_Parameters>
  </Observation_Area>
  <File_Area_Observational>
    <File>
      <file_name>CH2_OHR_NC_20230817T041219832_L1B.IMG</file_name>
      <file_size unit="byte">216000000</file_size>
      <md5_checksum>b4f7e29910d8ec889154a49c693a12cd</md5_checksum>
    </File>
    <Array_2D_Image>
      <offset unit="byte">0</offset>
      <axes>2</axes>
      <axis_index_order>Last_Index_Fastest</axis_index_order>
      <Element_Array>
        <data_type>SignedMSB2</data_type>
      </Element_Array>
      <Axis_Array>
        <axis_name>Line</axis_name>
        <elements>9000</elements>
        <sequence_number>1</sequence_number>
      </Axis_Array>
      <Axis_Array>
        <axis_name>Sample</axis_name>
        <elements>12000</elements>
        <sequence_number>2</sequence_number>
      </Axis_Array>
    </Array_2D_Image>
  </File_Area_Observational>
</Product_Observational>`,
    },
    tmc2: {
      productId: 'CH2_TMC_ST_20230817T041154010_L1B',
      targetName: 'Moon',
      missionPhase: 'Extended Science Orbit',
      instrumentId: 'TMC-2',
      instrumentName: 'Terrain Mapping Camera-2',
      acquisitionDate: '2023-08-17',
      acquisitionTime: '04:11:54.010 UTC',
      imageDimensions: { samples: 4000, lines: 16000 },
      resolution: { groundSampleDistance: 5.0, unit: 'm/pixel' },
      geometry: {
        subSpacecraftLatitude: -72.900,
        subSpacecraftLongitude: 53.250,
        spacecraftAltitudeKm: 99.4,
        centerLatitude: -72.9120,
        centerLongitude: 53.2800,
        incidenceAngleDeg: 43.5,
        emissionAngleDeg: 0.8,
        phaseAngleDeg: 43.6,
        solarAzimuthDeg: 124.8,
        solarElevationDeg: 46.5,
        slantDistanceKm: 99.41,
      },
      footprint: {
        cornerCoordinates: [
          [-72.750, 52.900],
          [-72.750, 53.650],
          [-73.080, 53.650],
          [-73.080, 52.900],
        ],
        areaKm2: 380.0,
        boundingPolygon: 'POLYGON((52.90 -72.75, 53.65 -72.75, 53.65 -73.08, 52.90 -73.08, 52.90 -72.75))',
      },
      radiometry: {
        exposureDurationMs: 8.2,
        calibrationVersion: 'ISRO-SAC-TMC2-CAL-v2.4',
        gainSetting: 'GAIN_NORM',
        unmodifiedRadiometry: true,
      },
      pdsXmlContent: `<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
  <Identification_Area>
    <logical_identifier>urn:isro:ch2:tmc2:data:ch2_tmc_st_20230817t041154010_l1b</logical_identifier>
    <version_id>2.0</version_id>
    <title>Chandrayaan-2 TMC-2 Nadir Strip Level-1B</title>
  </Identification_Area>
  <Observation_Area>
    <Time_Coordinates>
      <start_date_time>2023-08-17T04:11:54.010Z</start_date_time>
      <stop_date_time>2023-08-17T04:13:10.400Z</stop_date_time>
    </Time_Coordinates>
    <Observing_System>
      <Observing_System_Component>
        <name>Terrain Mapping Camera-2 (TMC-2)</name>
        <type>Stereo Optical Scanner</type>
      </Observing_System_Component>
    </Observing_System>
  </Observation_Area>
</Product_Observational>`,
    },
    iirs: {
      productId: 'CH2_IIR_HY_20230817T041140221_L1B',
      targetName: 'Moon',
      missionPhase: 'Extended Science Orbit',
      instrumentId: 'IIRS',
      instrumentName: 'Imaging Infrared Spectrometer',
      acquisitionDate: '2023-08-17',
      acquisitionTime: '04:11:40.221 UTC',
      imageDimensions: { samples: 250, lines: 1200 },
      resolution: { groundSampleDistance: 80.0, unit: 'm/pixel' },
      geometry: {
        subSpacecraftLatitude: -72.880,
        subSpacecraftLongitude: 53.200,
        spacecraftAltitudeKm: 99.5,
        centerLatitude: -72.9100,
        centerLongitude: 53.2750,
        incidenceAngleDeg: 44.1,
        emissionAngleDeg: 0.5,
        phaseAngleDeg: 44.2,
        solarAzimuthDeg: 125.1,
        solarElevationDeg: 45.9,
        slantDistanceKm: 99.51,
      },
      footprint: {
        cornerCoordinates: [
          [-72.600, 52.500],
          [-72.600, 54.100],
          [-73.220, 54.100],
          [-73.220, 52.500],
        ],
        areaKm2: 1240.0,
        boundingPolygon: 'POLYGON((52.50 -72.60, 54.10 -72.60, 54.10 -73.22, 52.50 -73.22, 52.50 -72.60))',
      },
      radiometry: {
        exposureDurationMs: 16.0,
        calibrationVersion: 'ISRO-SAC-IIRS-CUBE-v1.8',
        gainSetting: 'SPECTRAL_GAIN_HIGH_SWIR',
        unmodifiedRadiometry: true,
      },
      pdsXmlContent: `<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
  <Identification_Area>
    <logical_identifier>urn:isro:ch2:iirs:data:ch2_iir_hy_20230817t041140221_l1b</logical_identifier>
    <version_id>1.0</version_id>
    <title>Chandrayaan-2 IIRS Hyperspectral Radiance Cube Level-1B</title>
  </Identification_Area>
  <Observation_Area>
    <Time_Coordinates>
      <start_date_time>2023-08-17T04:11:40.221Z</start_date_time>
      <stop_date_time>2023-08-17T04:13:30.900Z</stop_date_time>
    </Time_Coordinates>
  </Observation_Area>
</Product_Observational>`,
    },
  },
  {
    id: 'shackleton-rim',
    name: 'Shackleton Crater Rim',
    region: 'South Pole Permanently Shadowed Region (89.9° S, 0.0° E)',
    description: 'Ultra-steep topography, permanent shadow traps, and peaks of eternal light presenting extreme illumination contrast and low-angle solar incidence.',
    targetFeature: 'Ridge Connecting Shackleton & de Gerlache',
    centerCoordinates: {
      lat: -89.72,
      lon: 114.3,
    },
    images: {
      ohrc: ohrcImg,
      tmc2: tmc2Img,
      iirs: iirsImg,
    },
    lroReference: {
      mission: 'NASA Lunar Reconnaissance Orbiter (LRO)',
      instrument: 'Narrow Angle Camera (LROC-NAC)',
      resolution: '0.50 m / px',
      sunAngle: 88.9,
      centerCoordinates: { lat: -89.72, lon: 114.28 },
      image: lroImg,
    },
    overlapStats: {
      commonAreaKm2: 7.91,
      overlapPercentageOHRC: 95.8,
      overlapPercentageTMC2: 38.6,
      overlapPercentageIIRS: 18.4,
      illuminationDiscrepancyDeg: 48.2,
      scaleDifferenceRatio: '1 : 20 : 320 (OHRC vs TMC-2 vs IIRS)',
    },
    ohrc: {
      productId: 'CH2_OHR_NC_20231104T182245109_L1B',
      targetName: 'Moon',
      missionPhase: 'Extended Science Orbit',
      instrumentId: 'OHRC',
      instrumentName: 'Orbiter High Resolution Camera',
      acquisitionDate: '2023-11-04',
      acquisitionTime: '18:22:45.109 UTC',
      imageDimensions: { samples: 12000, lines: 9000 },
      resolution: { groundSampleDistance: 0.25, unit: 'm/pixel' },
      geometry: {
        subSpacecraftLatitude: -89.72,
        subSpacecraftLongitude: 114.3,
        spacecraftAltitudeKm: 98.6,
        centerLatitude: -89.721,
        centerLongitude: 114.305,
        incidenceAngleDeg: 86.4,
        emissionAngleDeg: 2.1,
        phaseAngleDeg: 87.1,
        solarAzimuthDeg: 268.4,
        solarElevationDeg: 3.6,
        slantDistanceKm: 98.66,
      },
      footprint: {
        cornerCoordinates: [
          [-89.69, 114.10],
          [-89.69, 114.50],
          [-89.75, 114.50],
          [-89.75, 114.10],
        ],
        areaKm2: 8.25,
        boundingPolygon: 'POLYGON((114.10 -89.69, 114.50 -89.69, 114.50 -89.75, 114.10 -89.75, 114.10 -89.69))',
      },
      radiometry: {
        exposureDurationMs: 6.2,
        calibrationVersion: 'ISRO-SAC-CAL-v3.1',
        gainSetting: 'MAX_GAIN_CH01',
        unmodifiedRadiometry: true,
      },
      pdsXmlContent: `<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
  <Identification_Area>
    <logical_identifier>urn:isro:ch2:ohrc:data:ch2_ohr_nc_20231104t182245109_l1b</logical_identifier>
    <title>Chandrayaan-2 OHRC Shackleton Polar Scarp Observation</title>
  </Identification_Area>
</Product_Observational>`,
    },
    tmc2: {
      productId: 'CH2_TMC_ST_20231104T182210001_L1B',
      targetName: 'Moon',
      missionPhase: 'Extended Science Orbit',
      instrumentId: 'TMC-2',
      instrumentName: 'Terrain Mapping Camera-2',
      acquisitionDate: '2023-11-04',
      acquisitionTime: '18:22:10.001 UTC',
      imageDimensions: { samples: 4000, lines: 16000 },
      resolution: { groundSampleDistance: 5.0, unit: 'm/pixel' },
      geometry: {
        subSpacecraftLatitude: -89.70,
        subSpacecraftLongitude: 114.2,
        spacecraftAltitudeKm: 98.6,
        centerLatitude: -89.715,
        centerLongitude: 114.28,
        incidenceAngleDeg: 86.8,
        emissionAngleDeg: 1.4,
        phaseAngleDeg: 87.3,
        solarAzimuthDeg: 268.9,
        solarElevationDeg: 3.2,
        slantDistanceKm: 98.65,
      },
      footprint: {
        cornerCoordinates: [
          [-89.55, 113.5],
          [-89.55, 115.1],
          [-89.88, 115.1],
          [-89.88, 113.5],
        ],
        areaKm2: 365.0,
        boundingPolygon: 'POLYGON((113.5 -89.55, 115.1 -89.55, 115.1 -89.88, 113.5 -89.88, 113.5 -89.55))',
      },
      radiometry: {
        exposureDurationMs: 12.0,
        calibrationVersion: 'ISRO-SAC-TMC2-CAL-v2.4',
        gainSetting: 'HIGH_GAIN',
        unmodifiedRadiometry: true,
      },
      pdsXmlContent: `<Product_Observational><Title>TMC-2 Shackleton Stereo Coverage</Title></Product_Observational>`,
    },
    iirs: {
      productId: 'CH2_IIR_HY_20231104T182155099_L1B',
      targetName: 'Moon',
      missionPhase: 'Extended Science Orbit',
      instrumentId: 'IIRS',
      instrumentName: 'Imaging Infrared Spectrometer',
      acquisitionDate: '2023-11-04',
      acquisitionTime: '18:21:55.099 UTC',
      imageDimensions: { samples: 250, lines: 1200 },
      resolution: { groundSampleDistance: 80.0, unit: 'm/pixel' },
      geometry: {
        subSpacecraftLatitude: -89.68,
        subSpacecraftLongitude: 114.0,
        spacecraftAltitudeKm: 98.7,
        centerLatitude: -89.71,
        centerLongitude: 114.25,
        incidenceAngleDeg: 87.1,
        emissionAngleDeg: 0.9,
        phaseAngleDeg: 87.4,
        solarAzimuthDeg: 269.1,
        solarElevationDeg: 2.9,
        slantDistanceKm: 98.71,
      },
      footprint: {
        cornerCoordinates: [
          [-89.40, 112.8],
          [-89.40, 115.6],
          [-90.00, 115.6],
          [-90.00, 112.8],
        ],
        areaKm2: 1180.0,
        boundingPolygon: 'POLYGON((112.8 -89.40, 115.6 -89.40, 115.6 -90.00, 112.8 -90.00, 112.8 -89.40))',
      },
      radiometry: {
        exposureDurationMs: 24.0,
        calibrationVersion: 'ISRO-SAC-IIRS-CUBE-v1.8',
        gainSetting: 'MAX_GAIN_SWIR',
        unmodifiedRadiometry: true,
      },
      pdsXmlContent: `<Product_Observational><Title>IIRS Polar Volatiles Hyperspectral Cube</Title></Product_Observational>`,
    },
  },
  {
    id: 'tycho-central-peak',
    name: 'Tycho Central Peak Complex',
    region: 'Southern Lunar Highlands (43.3° S, 11.2° W)',
    description: 'Prominent young impact crater with sharp central mountain peak, massive impact melt ponds, and extensive ray system.',
    targetFeature: 'Central Peak Anorthositic Ridge',
    centerCoordinates: {
      lat: -43.31,
      lon: -11.22,
    },
    images: {
      ohrc: ohrcImg,
      tmc2: tmc2Img,
      iirs: iirsImg,
    },
    lroReference: {
      mission: 'NASA Lunar Reconnaissance Orbiter (LRO)',
      instrument: 'Narrow Angle Camera (LROC-NAC)',
      resolution: '0.50 m / px',
      sunAngle: 32.1,
      centerCoordinates: { lat: -43.31, lon: -11.22 },
      image: lroImg,
    },
    overlapStats: {
      commonAreaKm2: 8.52,
      overlapPercentageOHRC: 99.1,
      overlapPercentageTMC2: 44.3,
      overlapPercentageIIRS: 22.8,
      illuminationDiscrepancyDeg: 18.5,
      scaleDifferenceRatio: '1 : 20 : 320 (OHRC vs TMC-2 vs IIRS)',
    },
    ohrc: {
      productId: 'CH2_OHR_NC_20230612T114002340_L1B',
      targetName: 'Moon',
      missionPhase: 'Nominal Science Orbit',
      instrumentId: 'OHRC',
      instrumentName: 'Orbiter High Resolution Camera',
      acquisitionDate: '2023-06-12',
      acquisitionTime: '11:40:02.340 UTC',
      imageDimensions: { samples: 12000, lines: 9000 },
      resolution: { groundSampleDistance: 0.25, unit: 'm/pixel' },
      geometry: {
        subSpacecraftLatitude: -43.31,
        subSpacecraftLongitude: -11.22,
        spacecraftAltitudeKm: 100.2,
        centerLatitude: -43.3102,
        centerLongitude: -11.2214,
        incidenceAngleDeg: 31.4,
        emissionAngleDeg: 0.4,
        phaseAngleDeg: 31.6,
        solarAzimuthDeg: 88.2,
        solarElevationDeg: 58.6,
        slantDistanceKm: 100.21,
      },
      footprint: {
        cornerCoordinates: [
          [-43.28, -11.27],
          [-43.28, -11.17],
          [-43.34, -11.17],
          [-43.34, -11.27],
        ],
        areaKm2: 8.6,
        boundingPolygon: 'POLYGON((-11.27 -43.28, -11.17 -43.28, -11.17 -43.34, -11.27 -43.34, -11.27 -43.28))',
      },
      radiometry: {
        exposureDurationMs: 3.5,
        calibrationVersion: 'ISRO-SAC-CAL-v3.1',
        gainSetting: 'NORM_GAIN_CH01',
        unmodifiedRadiometry: true,
      },
      pdsXmlContent: `<Product_Observational><Title>Tycho Central Peak OHRC High-Res Scan</Title></Product_Observational>`,
    },
    tmc2: {
      productId: 'CH2_TMC_ST_20230612T113940100_L1B',
      targetName: 'Moon',
      missionPhase: 'Nominal Science Orbit',
      instrumentId: 'TMC-2',
      instrumentName: 'Terrain Mapping Camera-2',
      acquisitionDate: '2023-06-12',
      acquisitionTime: '11:39:40.100 UTC',
      imageDimensions: { samples: 4000, lines: 16000 },
      resolution: { groundSampleDistance: 5.0, unit: 'm/pixel' },
      geometry: {
        subSpacecraftLatitude: -43.30,
        subSpacecraftLongitude: -11.20,
        spacecraftAltitudeKm: 100.2,
        centerLatitude: -43.308,
        centerLongitude: -11.218,
        incidenceAngleDeg: 31.8,
        emissionAngleDeg: 0.3,
        phaseAngleDeg: 32.0,
        solarAzimuthDeg: 88.5,
        solarElevationDeg: 58.2,
        slantDistanceKm: 100.20,
      },
      footprint: {
        cornerCoordinates: [
          [-43.10, -11.55],
          [-43.10, -10.85],
          [-43.52, -10.85],
          [-43.52, -11.55],
        ],
        areaKm2: 390.0,
        boundingPolygon: 'POLYGON((-11.55 -43.10, -10.85 -43.10, -10.85 -43.52, -11.55 -43.52, -11.55 -43.10))',
      },
      radiometry: {
        exposureDurationMs: 6.8,
        calibrationVersion: 'ISRO-SAC-TMC2-CAL-v2.4',
        gainSetting: 'GAIN_NORM',
        unmodifiedRadiometry: true,
      },
      pdsXmlContent: `<Product_Observational><Title>Tycho TMC-2 Stereo DEM</Title></Product_Observational>`,
    },
    iirs: {
      productId: 'CH2_IIR_HY_20230612T113915880_L1B',
      targetName: 'Moon',
      missionPhase: 'Nominal Science Orbit',
      instrumentId: 'IIRS',
      instrumentName: 'Imaging Infrared Spectrometer',
      acquisitionDate: '2023-06-12',
      acquisitionTime: '11:39:15.880 UTC',
      imageDimensions: { samples: 250, lines: 1200 },
      resolution: { groundSampleDistance: 80.0, unit: 'm/pixel' },
      geometry: {
        subSpacecraftLatitude: -43.29,
        subSpacecraftLongitude: -11.18,
        spacecraftAltitudeKm: 100.3,
        centerLatitude: -43.305,
        centerLongitude: -11.215,
        incidenceAngleDeg: 32.2,
        emissionAngleDeg: 0.2,
        phaseAngleDeg: 32.3,
        solarAzimuthDeg: 88.7,
        solarElevationDeg: 57.8,
        slantDistanceKm: 100.31,
      },
      footprint: {
        cornerCoordinates: [
          [-42.90, -11.90],
          [-42.90, -10.45],
          [-43.72, -10.45],
          [-43.72, -11.90],
        ],
        areaKm2: 1290.0,
        boundingPolygon: 'POLYGON((-11.90 -42.90, -10.45 -42.90, -10.45 -43.72, -11.90 -43.72, -11.90 -42.90))',
      },
      radiometry: {
        exposureDurationMs: 14.5,
        calibrationVersion: 'ISRO-SAC-IIRS-CUBE-v1.8',
        gainSetting: 'NORM_GAIN_SWIR',
        unmodifiedRadiometry: true,
      },
      pdsXmlContent: `<Product_Observational><Title>Tycho IIRS Mineralogical Hyperspectral Cube</Title></Product_Observational>`,
    },
  },
];
