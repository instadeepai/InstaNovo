import csv
import json
import os
from importlib import resources
from pathlib import Path
from typing import List

import hydra
import numpy as np
import pytest
from typer import Typer
from typer.testing import CliRunner, Result

from instanovo.cli import combined_cli, instanovo_cli, instanovo_plus_cli

runner = CliRunner()


@pytest.fixture(scope="session")
def tmp_dir(tmp_path_factory: pytest.FixtureRequest) -> pytest.TempPathFactory:
    """Fixture for a temporary directory to store test files."""
    return tmp_path_factory.mktemp("test_data")


# Fixture for a dummy .mgf file
@pytest.fixture(scope="session")
def mgf_file(tmp_dir: Path) -> str:
    """Creates a temporary MGF (Mascot Generic Format) file with predefined content."""
    mgf_content = """BEGIN IONS
TITLE=0
PEPMASS=451.25348
CHARGE=2+
SCANS=F1:2478
RTINSECONDS=824.574
SEQ=IAHYNKR
63.994834899902344 0.0611930787563324
70.06543731689453 0.06860413402318954
84.081298828125 0.22455614805221558
85.08439636230469 0.06763620674610138
86.09666442871094 0.22344912588596344
110.07109069824219 0.3034861385822296
129.1020050048828 0.0932231917977333
138.06597900390625 0.07667151838541031
157.13291931152344 0.14716865122318268
175.1185302734375 0.19198034703731537
185.1283721923828 0.09717456996440887
209.10263061523438 0.13139843940734863
273.1337890625 0.09324286878108978
301.1282958984375 0.08515828102827072
303.21221923828125 0.07235292345285416
304.17529296875 0.07120858132839203
322.1859130859375 0.15834060311317444
350.6787414550781 0.07397215068340302
417.2552185058594 0.14982180297374725
580.3185424804688 0.31572264432907104
630.36572265625 0.06255878508090973
717.376708984375 0.5990896821022034
753.3748779296875 0.09976936876773834
788.4207763671875 0.35858696699142456
866.4544677734375 0.12016354501247406
END IONS
"""
    file_path = tmp_dir / "test.mgf"
    file_path.write_text(mgf_content)
    return str(file_path)


# Fixture for a dummy .mzML file
@pytest.fixture(scope="session")
def mzml_file(tmp_dir: Path) -> str:
    """Creates a temporary mzML file with predefined content."""
    mzml_content = """<?xml version="1.0" encoding="ISO-8859-1"?>
<indexedmzML xmlns="http://psi.hupo.org/schema_revision/mzML_0.99.10" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://psi.hupo.org/schema_revision/mzML_0.99.10_idx mzML0.99.10_idx.xsd">
  <mzML xmlns="http://psi.hupo.org/schema_revision/mzML_0.99.10" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://psi.hupo.org/schema_revision/mzML_0.99.10 mzML0.99.10.xsd" accession="test accession" id="test_id" version="test version">
    <cvList count="1">
      <cv id="MS" fullName="Proteomics Standards Initiative Mass Spectrometry Ontology" version="2.0.2" URI="http://psidev.sourceforge.net/ms/xml/mzdata/psi-ms.2.0.2.obo"/>
    </cvList>
    <fileDescription>
      <fileContent>
        <cvParam cvRef="MS" accession="MS:1000580" name="MSn spectrum" value=""/>
        <userParam name="number of cats" value="4" type=""/>
      </fileContent>
      <sourceFileList count="1">
        <sourceFile id="sf1" name="tiny1.RAW" location="file://F:/data/Exp01">
          <cvParam cvRef="MS" accession="MS:1000563" name="Xcalibur RAW file" value=""/>
          <cvParam cvRef="MS" accession="MS:1000569" name="SHA-1" value="71be39fb2700ab2f3c8b2234b91274968b6899b1"/>
        </sourceFile>
        <sourceFile id="sf2" name="parameters.par" location="file:///C:/settings/" />
      </sourceFileList>
      <contact>
        <cvParam cvRef="MS" accession="MS:1000586" name="contact name" value="William Pennington"/>
        <cvParam cvRef="MS" accession="MS:1000587" name="contact address" value="Higglesworth University, 12 Higglesworth Avenue, 12045, HI, USA"/>
        <cvParam cvRef="MS" accession="MS:1000588" name="contact URL" value="http://www.higglesworth.edu/"/>
        <cvParam cvRef="MS" accession="MS:1000589" name="contact email" value="wpennington@higglesworth.edu"/>
      </contact>
    </fileDescription>
    <referenceableParamGroupList count="2">
      <referenceableParamGroup id="CommonMS1SpectrumParams">
        <cvParam cvRef="MS" accession="MS:1000130" name="positive scan" value=""/>
        <cvParam cvRef="MS" accession="MS:1000498" name="full scan" value=""/>
      </referenceableParamGroup>
      <referenceableParamGroup id="CommonMS2SpectrumParams">
        <cvParam cvRef="MS" accession="MS:1000130" name="positive scan" value=""/>
        <cvParam cvRef="MS" accession="MS:1000498" name="full scan" value=""/>
      </referenceableParamGroup>
    </referenceableParamGroupList>
    <sampleList count="1">
      <sample id="sp1" name="Sample1">
      </sample>
    </sampleList>
    <instrumentConfigurationList count="1">
      <instrumentConfiguration id="LCQDeca">
        <cvParam cvRef="MS" accession="MS:1000554" name="LCQ Deca" value=""/>
        <cvParam cvRef="MS" accession="MS:1000529" name="instrument serial number" value="23433"/>
        <componentList count="3">
          <source order="1">
            <cvParam cvRef="MS" accession="MS:1000398" name="nanoelectrospray" value=""/>
          </source>
          <analyzer order="2">
            <cvParam cvRef="MS" accession="MS:1000082" name="quadrupole ion trap" value=""/>
          </analyzer>
          <detector order="3">
            <cvParam cvRef="MS" accession="MS:1000253" name="electron multiplier" value=""/>
          </detector>
        </componentList>
        <softwareRef ref="Xcalibur"/>
      </instrumentConfiguration>
    </instrumentConfigurationList>
    <softwareList count="3">
      <software id="Bioworks">
        <softwareParam cvRef="MS" accession="MS:1000533" name="Bioworks" version="3.3.1 sp1"/>
      </software>
      <software id="ReAdW">
        <softwareParam cvRef="MS" accession="MS:1000541" name="ReAdW" version="1.0"/>
      </software>
      <software id="Xcalibur">
        <softwareParam cvRef="MS" accession="MS:1000532" name="Xcalibur" version="2.0.5"/>
      </software>
    </softwareList>
    <dataProcessingList count="2">
      <dataProcessing id="XcaliburProcessing" softwareRef="Xcalibur">
        <processingMethod order="1">
          <cvParam cvRef="MS" accession="MS:1000033" name="deisotoping" value="false"/>
          <cvParam cvRef="MS" accession="MS:1000034" name="charge deconvolution" value="false"/>
          <cvParam cvRef="MS" accession="MS:1000035" name="peak picking" value="true"/>
        </processingMethod>
      </dataProcessing>
      <dataProcessing id="ReAdWConversion" softwareRef="ReAdW">
        <processingMethod order="2">
          <cvParam cvRef="MS" accession="MS:1000544" name="Conversion to mzML" value=""/>
        </processingMethod>
      </dataProcessing>
    </dataProcessingList>
    <acquisitionSettingsList count="1">
      <acquisitionSettings id="aS1" instrumentConfigurationRef="LCQDeca">
        <sourceFileRefList count="1">
          <sourceFileRef ref="sf2" />
        </sourceFileRefList>
        <targetList count="2">
          <target>
            <cvParam cvRef="MS" accession="MS:1000xxx" name="precursorMz" value="123.456" />
            <cvParam cvRef="MS" accession="MS:1000xxx" name="fragmentMz" value="456.789" />
            <cvParam cvRef="MS" accession="MS:1000xxx" name="dwell time" value="1" unitName="seconds" unitAccession="UO:0000010" />
            <cvParam cvRef="MS" accession="MS:1000xxx" name="active time" value="0.5" unitName="seconds" unitAccession="UO:0000010" />
          </target>
          <target>
            <cvParam cvRef="MS" accession="MS:1000xxx" name="precursorMz" value="231.673" />
            <cvParam cvRef="MS" accession="MS:1000xxx" name="fragmentMz" value="566.328" />
            <cvParam cvRef="MS" accession="MS:1000xxx" name="dwell time" value="1" unitName="seconds" unitAccession="UO:0000010" />
            <cvParam cvRef="MS" accession="MS:1000xxx" name="active time" value="0.5" unitName="seconds" unitAccession="UO:0000010" />
          </target>
        </targetList>
      </acquisitionSettings>
    </acquisitionSettingsList>
    <run id="Exp01" instrumentConfigurationRef="LCQDeca" sampleRef="sp1" startTimeStamp="2007-06-27T15:23:45.00035">
      <sourceFileRefList count="1">
        <sourceFileRef ref="sf1"/>
      </sourceFileRefList>
      <spectrumList count="2">
        <spectrum index="0" id="S19" nativeID="19" defaultArrayLength="10">
          <cvParam cvRef="MS" accession="MS:1000580" name="MSn spectrum" value=""/>
          <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>
          <spectrumDescription>
            <cvParam cvRef="MS" accession="MS:1000127" name="centroid mass spectrum" value=""/>
            <cvParam cvRef="MS" accession="MS:1000528" name="lowest m/z value" value="400.39"/>
            <cvParam cvRef="MS" accession="MS:1000527" name="highest m/z value" value="1795.56"/>
            <cvParam cvRef="MS" accession="MS:1000504" name="base peak m/z" value="445.347"/>
            <cvParam cvRef="MS" accession="MS:1000505" name="base peak intensity" value="120053"/>
            <cvParam cvRef="MS" accession="MS:1000285" name="total ion current" value="16675500"/>
            <scan instrumentConfigurationRef="LCQDeca">
              <referenceableParamGroupRef ref="CommonMS1SpectrumParams"/>
              <cvParam cvRef="MS" accession="MS:1000016" name="scan time" value="5.8905" unitAccession="MS:1000038" unitName="minute"/>
              <cvParam cvRef="MS" accession="MS:1000512" name="filter string" value="+ c NSI Full ms [ 400.00-1800.00]"/>
              <scanWindowList count="1">
                <scanWindow>
                  <cvParam cvRef="MS" accession="MS:1000501" name="scan m/z lower limit" value="400"/>
                  <cvParam cvRef="MS" accession="MS:1000500" name="scan m/z upper limit" value="1800"/>
                </scanWindow>
              </scanWindowList>
            </scan>
          </spectrumDescription>
          <binaryDataArrayList count="2">
            <binaryDataArray arrayLength="10" encodedLength="108" dataProcessingRef="XcaliburProcessing">
              <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
              <cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>
              <cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value=""/>
              <binary>AAAAAAAAAAAAAAAAAADwPwAAAAAAAABAAAAAAAAACEAAAAAAAAAQQAAAAAAAABRAAAAAAAAAGEAAAAAAAAAcQAAAAAAAACBAAAAAAAAAIkA=</binary>
            </binaryDataArray>
            <binaryDataArray arrayLength="10" encodedLength="108" dataProcessingRef="XcaliburProcessing">
              <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
              <cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>
              <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value=""/>
              <binary>AAAAAAAAJEAAAAAAAAAiQAAAAAAAACBAAAAAAAAAHEAAAAAAAAAYQAAAAAAAABRAAAAAAAAAEEAAAAAAAAAIQAAAAAAAAABAAAAAAAAA8D8=</binary>
            </binaryDataArray>
          </binaryDataArrayList>
        </spectrum>
        <spectrum index="1" id="S20" nativeID="20" defaultArrayLength="20">
          <cvParam cvRef="MS" accession="MS:1000580" name="MSn spectrum" value=""/>
          <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="2"/>
          <spectrumDescription>
            <cvParam cvRef="MS" accession="MS:1000127" name="centroid mass spectrum" value=""/>
            <cvParam cvRef="MS" accession="MS:1000528" name="lowest m/z value" value="320.39"/>
            <cvParam cvRef="MS" accession="MS:1000527" name="highest m/z value" value="1003.56"/>
            <cvParam cvRef="MS" accession="MS:1000504" name="base peak m/z" value="456.347"/>
            <cvParam cvRef="MS" accession="MS:1000505" name="base peak intensity" value="23433"/>
            <cvParam cvRef="MS" accession="MS:1000285" name="total ion current" value="16675500"/>
            <precursorList count="1">
              <precursor spectrumRef="S19">
                <isolationWindow>
                  <cvParam cvRef="MS" accession="MS:1000xxx" name="isolation center m/z" value="445.34"/>
                  <cvParam cvRef="MS" accession="MS:1000xxx" name="isolation half width" value="2.0"/>
                </isolationWindow>
                <selectedIonList count="1">
                  <selectedIon>
                    <cvParam cvRef="MS" accession="MS:1000040" name="m/z" value="445.34"/>
                    <cvParam cvRef="MS" accession="MS:1000041" name="charge state" value="2"/>
                  </selectedIon>
                </selectedIonList>
                <activation>
                  <cvParam cvRef="MS" accession="MS:1000133" name="collision-induced dissociation" value=""/>
                  <cvParam cvRef="MS" accession="MS:1000045" name="collision energy" value="35" unitAccession="MS:1000137" unitName="electron volt"/>
                </activation>
              </precursor>
            </precursorList>
            <scan instrumentConfigurationRef="LCQDeca">
              <referenceableParamGroupRef ref="CommonMS2SpectrumParams"/>
              <cvParam cvRef="MS" accession="MS:1000016" name="scan time" value="5.9905" unitAccession="MS:1000038" unitName="minute"/>
              <cvParam cvRef="MS" accession="MS:1000512" name="filter string" value="+ c d Full ms2  445.35@cid35.00 [ 110.00-905.00]"/>
              <scanWindowList count="1">
                <scanWindow>
                  <cvParam cvRef="MS" accession="MS:1000501" name="scan m/z lower limit" value="110"/>
                  <cvParam cvRef="MS" accession="MS:1000500" name="scan m/z upper limit" value="905"/>
                </scanWindow>
              </scanWindowList>
            </scan>
          </spectrumDescription>
          <binaryDataArrayList count="2">
            <binaryDataArray arrayLength="20" encodedLength="216" dataProcessingRef="XcaliburProcessing">
              <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
              <cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>
              <cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value=""/>
              <binary>AAAAAAAAAAAAAAAAAADwPwAAAAAAAABAAAAAAAAACEAAAAAAAAAQQAAAAAAAABRAAAAAAAAAGEAAAAAAAAAcQAAAAAAAACBAAAAAAAAAIkAAAAAAAAAkQAAAAAAAACZAAAAAAAAAKEAAAAAAAAAqQAAAAAAAACxAAAAAAAAALkAAAAAAAAAwQAAAAAAAADFAAAAAAAAAMkAAAAAAAAAzQA==</binary>
            </binaryDataArray>
            <binaryDataArray arrayLength="20" encodedLength="216" dataProcessingRef="XcaliburProcessing">
              <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
              <cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>
              <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value=""/>
              <binary>AAAAAAAANEAAAAAAAAAzQAAAAAAAADJAAAAAAAAAMUAAAAAAAAAwQAAAAAAAAC5AAAAAAAAALEAAAAAAAAAqQAAAAAAAAChAAAAAAAAAJkAAAAAAAAAkQAAAAAAAACJAAAAAAAAAIEAAAAAAAAAcQAAAAAAAABhAAAAAAAAAFEAAAAAAAAAQQAAAAAAAAAhAAAAAAAAAAEAAAAAAAADwPw==</binary>
            </binaryDataArray>
          </binaryDataArrayList>
        </spectrum>
      </spectrumList>
    </run>
  </mzML>
  <indexList count="1">
    <index name="spectrum">
      <offset idRef="S19" nativeID="19">4826</offset>
      <offset idRef="S20" nativeID="20">7576</offset>
    </index>
  </indexList>
  <indexListOffset>11300</indexListOffset>
  <fileChecksum>39112d76329487bf095cff1fec09ad871d8fedfb</fileChecksum>
</indexedmzML>
"""  # noqa: E501
    file_path = tmp_dir / "test.mzML"
    file_path.write_text(mzml_content)
    return str(file_path)


@pytest.mark.skip
@pytest.mark.parametrize("input_file_fixture", ["mgf_file", "mzml_file"])
def test_transformer_predict(
    tmp_dir: Path, input_file_fixture: str, request: pytest.FixtureRequest
) -> None:
    """Test the 'predict' command of the instanovo_cli with different input files."""
    # Get the actual file path from the fixture using request.getfixturevalue
    input_file = request.getfixturevalue(input_file_fixture)
    output_file = str(tmp_dir / "output.csv")

    result: Result = runner.invoke(
        instanovo_cli,
        [
            "predict",
            "--data-path",
            input_file,
            "--output-path",
            output_file,
            "device=cpu",
        ],
    )
    assert result.exit_code == 0

    with open(output_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames == [
            "scan_number",
            "precursor_mz",
            "precursor_charge",
            "experiment_name",
            "spectrum_id",
            "predictions",
            "predictions_tokenised",
            "log_probabilities",
            "token_log_probabilities",
            "delta_mass_ppm",
        ]
        assert len(rows) == 1
        data = rows[0]
        assert data["scan_number"] == "0"
        assert data["precursor_charge"] == "2"
        assert data["experiment_name"] == "test"
        assert data["spectrum_id"] == "test:0"
        if input_file_fixture == "mgf_file":
            assert data["precursor_mz"] == "451.25348"
            assert data["predictions"] == "LAHYNKK"
            assert data["predictions_tokenised"] == "L, A, H, Y, N, K, K"
            assert np.isclose(float(data["log_probabilities"]), -424.5889587402344, atol=0.1)
            import json

            token_log_probs = np.array(json.loads(data["token_log_probabilities"]))
            expected_token_log_probs = np.array(
                [
                    -0.5959059000015259,
                    -0.0059959776699543,
                    -0.01749008148908615,
                    -0.03598890081048012,
                    -0.48958998918533325,
                    -1.5242897272109985,
                    -0.656516432762146,
                ]
            )
            assert np.allclose(token_log_probs, expected_token_log_probs, atol=1e-2)
            assert np.isclose(float(data["delta_mass_ppm"]), 29919.088934228454, atol=1e-2)
        else:
            assert data["precursor_mz"] == "445.34"
            assert data["predictions"] == "HPASTGAAK"
            assert data["predictions_tokenised"] == "H, P, A, S, T, G, A, A, K"
            assert np.isclose(float(data["log_probabilities"]), -360.1124572753906, atol=0.1)
            import json

            token_log_probs = np.array(json.loads(data["token_log_probabilities"]))
            expected_token_log_probs = np.array(
                [
                    -0.49431324005126953,
                    -0.25507181882858276,
                    -0.09219300746917725,
                    -0.007722523063421249,
                    -0.593335747718811,
                    -0.23781751096248627,
                    -1.8838164806365967,
                    -2.089968204498291,
                    -0.1919376254081726,
                ]
            )
            assert np.allclose(token_log_probs, expected_token_log_probs, atol=1e-2)
            assert np.isclose(float(data["delta_mass_ppm"]), 55275.02020927855, atol=1e-2)


@pytest.mark.parametrize("cli", [combined_cli, instanovo_cli, instanovo_plus_cli])
def test_predict_nonexisting_config_name(cli: Typer) -> None:
    """Test the 'predict' command of the instanovo_cli with a nonexisting config name."""
    with pytest.raises(
        hydra.errors.MissingConfigException,
        match="Cannot find primary config 'nonexisting'. "
        "Check that it's in your config search path.",
    ):
        runner.invoke(cli, ["predict", "--config-name", "nonexisting"], catch_exceptions=False)


@pytest.mark.parametrize("cli", [combined_cli, instanovo_cli, instanovo_plus_cli])
def test_predict_nonexisting_config_path(cli: Typer) -> None:
    """Test the 'predict' command of the instanovo_cli with a nonexisting config path."""
    with pytest.raises(
        hydra.errors.MissingConfigException, match="Primary config directory not found."
    ):
        runner.invoke(cli, ["predict", "--config-path", "nonexisting"], catch_exceptions=False)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], "Run predictions with InstaNovo and optionally with InstaNovo+."),
        (["--help"], "Run predictions with InstaNovo and optionally with InstaNovo+."),
        (
            ["transformer", "--help"],
            "Run predictions or train with only the transformer-based InstaNovo model.",
        ),
        (
            ["diffusion", "--help"],
            "Run predictions or train with only the diffusion-based InstaNovo+ model.",
        ),
    ],
)
def test_main_help(args: List[str], expected: str) -> None:
    """Test the main command help text is displayed."""
    result: Result = runner.invoke(combined_cli, args)
    assert result.exit_code == 0
    assert expected in result.stdout


def test_predict_transformer(caplog: pytest.FixtureRequest) -> None:
    """Test the 'predict' command of the instanovo_cli."""
    with caplog.at_level("INFO"):
        result: Result = runner.invoke(
            combined_cli, ["transformer", "predict", "--config-name", "unit_test"]
        )
        assert result.exit_code == 0
        assert (
            "data_path: ./tests/instanovo_test_resources/example_data/test_sample.mgf"
            in caplog.text
        )
        assert "max_charge: 3" in caplog.text
        assert "denovo: false" in caplog.text

        assert "instanovo_model: ./tests/instanovo_test_resources/model.ckpt" in caplog.text
        assert "output_path: ./tests/instanovo_test_resources/test_sample_preds.csv" in caplog.text
        assert "knapsack_path: ./tests/instanovo_test_resources/example_knapsack" in caplog.text
        assert "use_knapsack: false" in caplog.text
        assert "num_beams: 5" in caplog.text
        assert "max_length: 40" in caplog.text


# TODO until a diffusion checkpoint is available, skip diffusion tests
def is_diffusion_checkpoint_available() -> bool:
    """Check if the diffusion checkpoint is available."""
    with resources.files("instanovo").joinpath("models.json").open("r", encoding="utf-8") as f:
        models_config = json.load(f)
    try:
        model_info = models_config["diffusion"]["instanovo-plus-v1.1.0"]
        if "local" in model_info:
            instanovo_plus_model = model_info["local"]
            return os.path.isdir(instanovo_plus_model)
        return False
    except KeyError:
        return False


@pytest.mark.skipif(
    not is_diffusion_checkpoint_available(), reason="no diffusion checkpoint available"
)
def test_predict_diffusion(caplog: pytest.FixtureRequest) -> None:
    """Test the 'diffusion predict' command of the instanovo_cli."""
    with caplog.at_level("INFO"):
        result: Result = runner.invoke(
            combined_cli, ["diffusion", "predict", "--config-name", "instanovoplus_unit_test"]
        )
        assert result.exit_code == 0
        assert (
            "data_path: ./tests/instanovo_test_resources/example_data/test_sample.mgf"
            in caplog.text
        )
        assert "max_charge: 3" in caplog.text
        assert "denovo: false" in caplog.text
        assert "instanovo_plus_model: ./tests/instanovo_test_resources/instanovoplus" in caplog.text
        assert (
            "output_path: ./tests/instanovo_test_resources/instanovoplus/test_sample_preds.csv"
            in caplog.text
        )
        assert "knapsack_path: null" in caplog.text
        assert "max_length: 6" in caplog.text


@pytest.mark.skipif(
    not is_diffusion_checkpoint_available(), reason="no diffusion checkpoint available"
)
def test_predict(caplog: pytest.FixtureRequest) -> None:
    """Test the 'predict' command of the instanovo_cli."""
    with caplog.at_level("INFO"):
        result: Result = runner.invoke(
            combined_cli,
            ["predict", "--config-name", "unit_test"],
        )
        assert result.exit_code == 0
        assert (
            "data_path: ./tests/instanovo_test_resources/example_data/test_sample.mgf"
            in caplog.text
        )
        assert "max_charge: 3" in caplog.text
        assert "denovo: false" in caplog.text

        assert "instanovo_model: ./tests/instanovo_test_resources/model.ckpt" in caplog.text
        assert "output_path: ./tests/instanovo_test_resources/test_sample_preds.csv" in caplog.text
        assert "knapsack_path: ./tests/instanovo_test_resources/example_knapsack" in caplog.text
        assert "use_knapsack: false" in caplog.text
        assert "num_beams: 5" in caplog.text
        assert "max_length: 40" in caplog.text


@pytest.mark.parametrize("input_file_fixture", ["mgf_file", "mzml_file"])
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            [
                "predict",
                "--instanovo-model",
                "nonexisting",
                "--output-path",
                "predictions.csv",
            ],
            "InstaNovo",
        ),
        # (["predict", "--instanovo-plus-model", "nonexisting"], "InstaNovo+"), #TODO
        (
            [
                "predict",
                "--instanovo-model",
                "nonexisting",
                "--instanovo-plus-model",
                "nonexisting",
                "--output-path",
                "predictions.csv",
            ],
            "InstaNovo",
        ),
        (
            [
                "transformer",
                "predict",
                "--instanovo-model",
                "nonexisting",
                "--output-path",
                "predictions.csv",
            ],
            "InstaNovo",
        ),
        #    (["diffusion", "predict", "--instanovo-plus-model", "nonexisting"], "InstaNovo+") #TODO
    ],
)
def test_model(
    input_file_fixture: str,
    request: pytest.FixtureRequest,
    args: List[str],
    expected: str,
) -> None:
    """Test the 'predict' command of the instanovo_cli with a nonexisting instanovo model."""
    with pytest.raises(ValueError, match=f"{expected} model ID 'nonexisting' is not supported."):
        runner.invoke(
            combined_cli,
            args + ["--data-path", request.getfixturevalue(input_file_fixture)],
            catch_exceptions=False,
        )


@pytest.mark.parametrize("input_file_fixture", ["mgf_file", "mzml_file"])
@pytest.mark.parametrize(
    "args",
    [
        [
            "predict",
            "--instanovo-model",
            "checkpoint.invalid",
            "--output-path",
            "predictions.csv",
        ],
        [
            "transformer",
            "predict",
            "--instanovo-model",
            "checkpoint.invalid",
            "--output-path",
            "predictions.csv",
        ],
    ],
)
def test_instanovo_model_suffix(
    input_file_fixture: str, request: pytest.FixtureRequest, args: List[str]
) -> None:
    """Test the 'predict' command of the instanovo cli with a file with a non-supported suffix."""
    with runner.isolated_filesystem():
        with open("checkpoint.invalid", "w") as f:
            f.write("dummy checkpoint file")
        with pytest.raises(
            ValueError,
            match="Checkpoint file 'checkpoint.invalid' should end with extension '.ckpt'.",
        ):
            runner.invoke(
                combined_cli,
                args + ["--data-path", request.getfixturevalue(input_file_fixture)],
                catch_exceptions=False,
            )


@pytest.mark.parametrize("input_file_fixture", ["mgf_file", "mzml_file"])
@pytest.mark.parametrize(
    ("extension", "expected"),
    [
        (".ckpt", r"\*.yaml, \*.pt"),
        (".yaml", r"\*.ckpt, \*.pt"),
        (".pt", r"\*.ckpt, \*.yaml"),
        (".txt", r"\*.ckpt, \*.yaml, \*.pt"),
    ],
)
@pytest.mark.parametrize(
    "args",
    [
        # ["predict", "--instanovo-plus-model", "checkpoint_dir"], #TODO
        [
            "diffusion",
            "predict",
            "--instanovo-plus-model",
            "checkpoint_dir",
            "--output-path",
            "predictions.csv",
        ]
    ],
)
def test_instanovoplus_model(
    input_file_fixture: str,
    request: pytest.FixtureRequest,
    args: List[str],
    extension: str,
    expected: str,
) -> None:
    """Test the 'predict' command of the instanovo plus cli with a non existing directory."""
    with runner.isolated_filesystem():
        os.mkdir("checkpoint_dir")
        with open(f"checkpoint_dir/file.{extension}", "w") as f:
            f.write("dummy checkpoint file")

        with pytest.raises(
            ValueError,
            match=(
                r"The directory 'checkpoint_dir' is missing the following required file\(s\): "
                f"{expected}."
            ),
        ):
            runner.invoke(
                combined_cli,
                args + ["--data-path", request.getfixturevalue(input_file_fixture)],
                catch_exceptions=False,
            )


@pytest.mark.parametrize(
    "args",
    [
        ["predict", "--data-path", "*.mgf"],
        ["transformer", "predict", "--data-path", "*.mgf"],
        ["diffusion", "predict", "--data-path", "*.mgf"],
    ],
)
def test_nonexisting_data_path(args: List[str]) -> None:
    """Test the 'predict' command of the instanovo_cli with a nonexisting data path."""
    with runner.isolated_filesystem():
        with pytest.raises(
            ValueError,
            match=r"The data_path '\*.mgf' doesn't correspond to any file\(s\).",
        ):
            runner.invoke(combined_cli, args, catch_exceptions=False)


@pytest.mark.parametrize(
    "args",
    [
        ["predict", "--config-name", "default"],
        ["transformer", "predict", "--config-name", "default"],
        ["diffusion", "predict", "--config-name", "default"],
    ],
)
def test_no_data_path(args: List[str]) -> None:
    """Test the 'predict' command of the instanovo_cli with no data path."""
    with runner.isolated_filesystem():
        with pytest.raises(
            ValueError,
            match=(
                r"Expected 'data_path' but found None. Please specify it in the "
                r"`config/inference/<your_config>.yaml` configuration file or with the cli flag "
                r"`--data-path='path/to/data'`. Allows `.mgf`, `.mzml`, `.mzxml`, a directory, or a"
                r" `.parquet` file. Glob notation is supported:  eg.: "
                r"`--data-path='./experiment/\*.mgf'`."
            ),
        ):
            runner.invoke(combined_cli, args, catch_exceptions=False)


@pytest.mark.parametrize("input_file_fixture", ["mgf_file", "mzml_file"])
@pytest.mark.parametrize(
    "args",
    [
        ["predict", "--config-name", "default"],
        ["transformer", "predict", "--config-name", "default"],
        ["diffusion", "predict", "--config-name", "default"],
    ],
)
def test_no_output_path(
    input_file_fixture: str, request: pytest.FixtureRequest, args: List[str]
) -> None:
    """Test the 'predict' command of the instanovo_cli with no output path."""
    with runner.isolated_filesystem():
        with pytest.raises(
            ValueError,
            match=(
                r"Expected 'output_path' but found None. Please specify it in the "
                r"`config/inference/<your_config>.yaml` configuration file or with the cli flag "
                r"`--output-path=path/to/output_file`."
            ),
        ):
            runner.invoke(
                combined_cli,
                args + ["--data-path", request.getfixturevalue(input_file_fixture)],
                catch_exceptions=False,
            )


@pytest.mark.parametrize(
    "args",
    [
        # ["predict", "--config-name", "unit_test"], # TODO
        ["transformer", "predict", "--config-name", "unit_test"],
    ],
)
def test_beams(args: List[str], caplog: pytest.FixtureRequest) -> None:
    """Test beams."""
    with caplog.at_level("INFO"):
        runner.invoke(
            combined_cli,
            args + ["save_beams=true", "num_beams=1"],
            catch_exceptions=False,
        )
        assert (
            "num_beams is 1 and will override save_beams. Only use save_beams in beam search."
            in caplog.text
        )


@pytest.mark.parametrize("model", ["transformer", "diffusion"])
def test_n_gpu(model: str) -> None:
    """Test with n_gpu that is not supported."""
    with pytest.raises(ValueError, match="n_gpu > 1 currently not supported."):
        runner.invoke(
            combined_cli,
            [model, "train", "--config-name", "instanovo_unit_test", "n_gpu=10"],
            catch_exceptions=False,
        )


@pytest.mark.parametrize(
    ("args", "name"),
    [
        (["transformer", "train", "--config-name", "instanovo_unit_test"], "InstaNovo"),
        (
            ["diffusion", "train", "--config-name", "instanovoplus_unit_test"],
            "InstaNovo+",
        ),
    ],
)
def test_train(args: List[str], name: str, caplog: pytest.FixtureRequest) -> None:
    """Test the 'train' command of the combined_cli."""
    with caplog.at_level("INFO"):
        result: Result = runner.invoke(combined_cli, args)
        assert result.exit_code == 0
        assert f"Initializing {name} training." in caplog.text
        assert f"{name} training started" in caplog.text
        assert "Python version: " in caplog.text
        assert "PyTorch version: " in caplog.text
        assert "CUDA version: " in caplog.text
        assert f"{name} training config:" in caplog.text
        assert f"{name} training finished" in caplog.text


def test_version_command() -> None:
    """Test the version command."""
    result: Result = runner.invoke(combined_cli, ["version"])
    assert result.exit_code == 0
    assert "InstaNovo" in result.stdout
    assert "InstaNovo+" in result.stdout
    assert "PyTorch" in result.stdout
    assert "Lightning" in result.stdout
