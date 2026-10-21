$ErrorActionPreference = 'Stop'
$root = (Get-Location).Path
$outDir = Join-Path $root 'model_training\outputs\percentile_comparison_experiment'
$reportPath = Join-Path $outDir 'ZeroShield_AI_Model_Comparison_Report.docx'
$master = @(Import-Csv (Join-Path $outDir 'master_results.csv'))
$thresholds = @(Import-Csv (Join-Path $outDir 'threshold_results.csv'))
$categories = @(Import-Csv (Join-Path $outDir 'zero_day_category_results.csv'))
$models = @(Import-Csv (Join-Path $outDir 'model_summary.csv'))
$bestPoints = @(Import-Csv (Join-Path $outDir 'best_operating_points.csv'))
$config = Get-Content (Join-Path $outDir 'experiment_config.json') -Raw | ConvertFrom-Json
$featureList = @($config.features)
$modelNames = @('AE','VAE','DAE','CAE','Deep SVDD','OCSVM')
$modelLabels = @{
    'AE'='Autoencoder'; 'VAE'='Variational Autoencoder'; 'DAE'='Denoising Autoencoder'
    'CAE'='Contractive Autoencoder'; 'Deep SVDD'='Deep SVDD'; 'OCSVM'='One-Class SVM'
}
$wdAlignCenter = 1
$wdAlignLeft = 0
$wdSectionBreakNextPage = 2
$wdStatisticPages = 2
$word = $null
$doc = $null

function Fmt-Num([object]$v, [int]$digits = 4) {
    return ('{0:F' + $digits + '}') -f [double]$v
}
function Fmt-Pct([object]$v, [int]$digits = 2) {
    return ((Fmt-Num ([double]$v * 100) $digits) + '%')
}
function Fmt-Count([object]$v) { return ('{0:N0}' -f [double]$v) }
function Move-End {
    [void]$word.Selection.EndKey(6)
}
function Add-Paragraph([string]$text, [string]$style = 'Normal', [switch]$Center, [switch]$Italic, [switch]$Bold) {
    Move-End
    $sel = $word.Selection
    $sel.Style = $style
    if ($Center) { $sel.ParagraphFormat.Alignment = $wdAlignCenter }
    else { $sel.ParagraphFormat.Alignment = $wdAlignLeft }
    if ($style -eq 'Normal') {
        $sel.Font.Name = 'Aptos'
        $sel.Font.Size = 10
        $sel.ParagraphFormat.SpaceAfter = 6
        $sel.ParagraphFormat.LineSpacingRule = 0
    }
    $sel.Font.Italic = [bool]$Italic
    $sel.Font.Bold = [bool]$Bold
    $sel.TypeText($text)
    $sel.TypeParagraph()
}
function Add-Heading([string]$text, [int]$level = 1) {
    Add-Paragraph $text ("Heading $level")
}
function Add-TableCaption([string]$text) {
    Add-Paragraph $text 'Caption' -Bold
}
function Add-Table([string[]]$Headers, [System.Collections.IEnumerable]$Rows, [int]$FontSize = 8) {
    Move-End
    $rowArray = @($Rows)
    $sel = $word.Selection
    $table = $doc.Tables.Add($sel.Range, $rowArray.Count + 1, $Headers.Count)
    $table.Style = 'Table Grid'
    $table.AllowAutoFit = $true
    $table.AutoFitBehavior(2)
    $table.Rows.Item(1).HeadingFormat = -1
    $table.Rows.Item(1).Range.Font.Bold = $true
    $table.Rows.Item(1).Range.Font.Color = 16777215
    $table.Rows.Item(1).Shading.BackgroundPatternColor = 6579300
    $table.Range.Font.Name = 'Aptos'
    $table.Range.Font.Size = $FontSize
    $table.Range.ParagraphFormat.SpaceAfter = 0
    $table.Range.ParagraphFormat.SpaceBefore = 0
    $table.Range.Cells.VerticalAlignment = 1
    for ($c = 0; $c -lt $Headers.Count; $c++) {
        $table.Cell(1, $c + 1).Range.Text = [string]$Headers[$c]
    }
    for ($r = 0; $r -lt $rowArray.Count; $r++) {
        $values = @($rowArray[$r])
        for ($c = 0; $c -lt $Headers.Count; $c++) {
            $val = if ($c -lt $values.Count -and $null -ne $values[$c]) { [string]$values[$c] } else { '' }
            $table.Cell($r + 2, $c + 1).Range.Text = $val
        }
    }
    Move-End
    $word.Selection.TypeParagraph()
}
function Start-Landscape {
    Move-End
    $word.Selection.InsertBreak($wdSectionBreakNextPage)
    $sec = $doc.Sections.Item($doc.Sections.Count)
    $sec.PageSetup.Orientation = 1
    $sec.PageSetup.PaperSize = 7
    $sec.PageSetup.TopMargin = $word.CentimetersToPoints(1.4)
    $sec.PageSetup.BottomMargin = $word.CentimetersToPoints(1.4)
    $sec.PageSetup.LeftMargin = $word.CentimetersToPoints(1.4)
    $sec.PageSetup.RightMargin = $word.CentimetersToPoints(1.4)
}
function End-Landscape {
    Move-End
    $word.Selection.InsertBreak($wdSectionBreakNextPage)
    $sec = $doc.Sections.Item($doc.Sections.Count)
    $sec.PageSetup.Orientation = 0
    $sec.PageSetup.PaperSize = 7
    $sec.PageSetup.TopMargin = $word.CentimetersToPoints(1.9)
    $sec.PageSetup.BottomMargin = $word.CentimetersToPoints(1.9)
    $sec.PageSetup.LeftMargin = $word.CentimetersToPoints(2.0)
    $sec.PageSetup.RightMargin = $word.CentimetersToPoints(2.0)
}
function Add-Figure([int]$number, [string]$filename, [string]$caption, [string]$interpretation) {
    Move-End
    $sel = $word.Selection
    $sel.ParagraphFormat.Alignment = $wdAlignCenter
    $imgPath = Join-Path $outDir $filename
    if (-not (Test-Path -LiteralPath $imgPath)) { throw "Missing figure: $imgPath" }
    $shape = $sel.InlineShapes.AddPicture($imgPath, $false, $true, $sel.Range)
    $shape.LockAspectRatio = -1
    $shape.Width = $word.CentimetersToPoints(16.1)
    $shape.Range.ParagraphFormat.Alignment = $wdAlignCenter
    $sel.SetRange($shape.Range.End, $shape.Range.End)
    $sel.TypeParagraph()
    Add-Paragraph ("Figure {0}. {1}" -f $number, $caption) 'Caption' -Center -Italic
    Add-Paragraph ("Interpretation. {0}" -f $interpretation)
}
function Median([double[]]$values) {
    $v = @($values | Sort-Object)
    if ($v.Count % 2) { return [double]$v[[int][math]::Floor($v.Count / 2)] }
    return (([double]$v[$v.Count / 2 - 1] + [double]$v[$v.Count / 2]) / 2)
}
function Best-Row([string]$field, [switch]$Ascending) {
    if ($Ascending) { return $master | Sort-Object { [double]$_.$field } | Select-Object -First 1 }
    return $master | Sort-Object { [double]$_.$field } -Descending | Select-Object -First 1
}

if ($master.Count -ne 30 -or $categories.Count -ne 270 -or $featureList.Count -ne 43) {
    throw "Input validation failed: master=$($master.Count), categories=$($categories.Count), features=$($featureList.Count)."
}
if (Test-Path -LiteralPath $reportPath) { throw "Refusing to overwrite existing report: $reportPath" }

$bestF1 = Best-Row 'F1'
$bestZD = Best-Row 'Zero_Day_Detection_Rate'
$bestRecall = Best-Row 'Recall'
$bestPrecision = Best-Row 'Precision'
$bestFpr = Best-Row 'FPR' -Ascending
$balancedScores = foreach ($row in $master) {
    [pscustomobject]@{ Row = $row; Score = [math]::Pow([double]$row.F1 * [double]$row.Zero_Day_Detection_Rate * [double]$row.Specificity, 1.0/3.0) }
}
$bestBalanced = $balancedScores | Sort-Object Score -Descending | Select-Object -First 1
$at95 = @($master | Where-Object { [int]$_.Percentile -eq 95 } | Sort-Object { [array]::IndexOf($modelNames, $_.Model) })
$medians = @{}
foreach ($p in @(90,93,95,97,99)) {
    $d = @($master | Where-Object { [int]$_.Percentile -eq $p })
    $medians[$p] = @{
        FPR = Median ([double[]]@($d | ForEach-Object { [double]$_.FPR }))
        Recall = Median ([double[]]@($d | ForEach-Object { [double]$_.Recall }))
        F1 = Median ([double[]]@($d | ForEach-Object { [double]$_.F1 }))
        Precision = Median ([double[]]@($d | ForEach-Object { [double]$_.Precision }))
        ZD = Median ([double[]]@($d | ForEach-Object { [double]$_.Zero_Day_Detection_Rate }))
        Accuracy = Median ([double[]]@($d | ForEach-Object { [double]$_.Accuracy }))
    }
}
$masterNumeric = $master | ForEach-Object { $_ }
$best95F1 = $at95 | Sort-Object { [double]$_.F1 } -Descending | Select-Object -First 1
$best95ZD = $at95 | Sort-Object { [double]$_.Zero_Day_Detection_Rate } -Descending | Select-Object -First 1
$best95Fpr = $at95 | Sort-Object { [double]$_.FPR } | Select-Object -First 1
$positiveCount = [long]$master[0].Evaluation_Positive_Samples
$negativeCount = [long]$master[0].Evaluation_Negative_Samples
$validationN = [long]$master[0].Validation_Rows
$zeroN = [long]$master[0].Zero_Day_Samples
$trainN = [long]$config.sample_sizes.'neural and Deep SVDD training'
$referenceN = [long]$config.sample_sizes.'threshold reference'
$startText = ([datetime]$config.start_time_utc).ToString('yyyy-MM-dd HH:mm UTC')
$endText = ([datetime]$config.end_time_utc).ToString('yyyy-MM-dd HH:mm UTC')
$durationMin = [math]::Round([double]$config.experiment_duration_seconds / 60, 1)

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.Options.CheckSpellingAsYouType = $false
    $word.Options.CheckGrammarAsYouType = $false
    $doc = $word.Documents.Add()
    $doc.PageSetup.PaperSize = 7
    $doc.PageSetup.TopMargin = $word.CentimetersToPoints(1.9)
    $doc.PageSetup.BottomMargin = $word.CentimetersToPoints(1.9)
    $doc.PageSetup.LeftMargin = $word.CentimetersToPoints(2.0)
    $doc.PageSetup.RightMargin = $word.CentimetersToPoints(2.0)
    $doc.Styles.Item('Normal').Font.Name = 'Aptos'
    $doc.Styles.Item('Normal').Font.Size = 10
    $doc.Styles.Item('Normal').ParagraphFormat.SpaceAfter = 6
    $doc.Styles.Item('Heading 1').Font.Name = 'Aptos Display'
    $doc.Styles.Item('Heading 1').Font.Size = 17
    $doc.Styles.Item('Heading 1').Font.Color = 6579300
    $doc.Styles.Item('Heading 2').Font.Name = 'Aptos Display'
    $doc.Styles.Item('Heading 2').Font.Size = 12
    $doc.Styles.Item('Heading 2').Font.Color = 6579300
    $doc.Styles.Item('Caption').Font.Name = 'Aptos'
    $doc.Styles.Item('Caption').Font.Size = 8.5
    $doc.Styles.Item('Caption').Font.Color = 6579300
    # Title page
    Move-End
    $sel = $word.Selection
    $sel.ParagraphFormat.Alignment = $wdAlignCenter
    $sel.TypeParagraph(); $sel.TypeParagraph(); $sel.TypeParagraph()
    $sel.Style = 'Title'; $sel.Font.Name = 'Aptos Display'; $sel.Font.Size = 27; $sel.Font.Bold = $true
    $sel.TypeText('ZeroShield AI Model Comparison Report'); $sel.TypeParagraph()
    $sel.Font.Size = 15; $sel.Font.Bold = $false
    $sel.TypeText('A reproducible six-model benign-reference threshold experiment'); $sel.TypeParagraph(); $sel.TypeParagraph()
    $sel.Style = 'Normal'; $sel.Font.Size = 11
    $sel.TypeText('Autoencoder | Variational Autoencoder | Denoising Autoencoder | Contractive Autoencoder | Deep SVDD | One-Class SVM'); $sel.TypeParagraph(); $sel.TypeParagraph()
    $sel.TypeText(('Final Year Research Experiment  |  {0}' -f (Get-Date -Format 'dd MMMM yyyy'))); $sel.TypeParagraph(); $sel.TypeParagraph()
    $sel.TypeText('Primary comparison: the 95th percentile, with 90th, 93rd, 97th, and 99th percentile operating points reported alongside it.'); $sel.TypeParagraph()
    $sel.TypeText(('Evaluation: {0} validation rows and {1} zero-day attack rows, streamed in chunks.' -f (Fmt-Count $validationN), (Fmt-Count $zeroN))); $sel.TypeParagraph()
    $sel.TypeText(('All six models completed. Deep SVDD at the 95th percentile reached {0} F1 and {1} zero-day detection on this evaluation split.' -f (Fmt-Num ([double]$best95F1.F1) 4), (Fmt-Pct $best95ZD.Zero_Day_Detection_Rate))); $sel.TypeParagraph()
    Move-End; $word.Selection.InsertBreak(7)

    Add-Heading 'Contents' 1
    $contents = @(
        '1 Objective','2 Experimental setup','3 Dataset','4 Preprocessing','5 Feature engineering and feature order',
        '6 Models','7 Training configuration','8 Threshold methodology','9 Evaluation metrics','10 Experimental results',
        '11 Percentile comparison','12 Zero-day evaluation','13 Visualizations','14 Model-by-model analysis',
        '15 Cross-model comparison','16 Threshold trade-offs','17 Best operating points','18 Limitations',
        '19 Research observations','20 Conclusion','21 Reproducibility information','22 Appendix'
    )
    foreach ($item in $contents) { Add-Paragraph $item }
    Move-End; $word.Selection.InsertBreak(7)

    Add-Heading '1 Objective' 1
    Add-Paragraph 'This experiment compares six existing ZeroShield anomaly detectors on the project''s prepared 43-feature CIC-IoT-2023-derived representation. Each detector is fitted on benign traffic only. Five decision thresholds per model are calibrated from a disjoint benign reference set, then frozen and applied to the same full validation and zero-day files. The intended result is a reproducible comparison of detection performance and false-alarm trade-offs, including category-level zero-day results.'

    Add-Heading '2 Experimental setup' 1
    Add-Paragraph 'The six implementations in the repository-root models.py were used: AE, VAE, DAE, CAE, Deep SVDD, and RBF One-Class SVM. Neural models and Deep SVDD share the same 500,000 benign training rows; OCSVM uses a seeded 10,000-row subset to limit kernel cost. All models receive the same ordered input features and prepared values. The same model instance is scored at all five percentiles.'
    Add-Paragraph ('Random seed: {0}. Device: CPU ({1} logical processors). Runtime: Python {2}; PyTorch {3}; scikit-learn {4}; pandas {5}; NumPy {6}.' -f $config.seed, $config.environment.logical_cpus, $config.environment.python, $config.environment.torch, $config.environment.scikit_learn, $config.environment.pandas, $config.environment.numpy)

    Add-Heading '3 Dataset' 1
    Add-Paragraph 'The file inventory was checked before training. The project contains a benign-only prepared training file, a mixed benign/known-attack validation file, and the established attack-only zero-day file. No separate calibration CSV was found; a disjoint benign calibration subset was therefore reserved from the benign-only training source.'
    Add-TableCaption 'Table 1. Dataset summary and experiment split.'
    $datasetRows = [System.Collections.Generic.List[object[]]]::new()
    $datasetRows.Add([object[]]@('Benign source','train_benign_only.csv', (Fmt-Count $config.datasets.training_source_rows_scanned), 'Source scanned; 600,000-row sample drawn with seed 42.'))
    $datasetRows.Add([object[]]@('Model fitting','Disjoint subset of benign source', (Fmt-Count $trainN), 'Used by AE, VAE, DAE, CAE, and Deep SVDD.'))
    $datasetRows.Add([object[]]@('Threshold reference','Disjoint subset of benign source', (Fmt-Count $referenceN), 'Benign scores only; not used for fitting.'))
    $datasetRows.Add([object[]]@('Validation evaluation','validation_set.csv', (Fmt-Count $validationN), ('{0} attacks and {1} benign rows.' -f (Fmt-Count $positiveCount), (Fmt-Count $negativeCount))))
    $datasetRows.Add([object[]]@('Zero-day evaluation','zero_day_test_scaled_v2.csv', (Fmt-Count $zeroN), 'Attack-only; nine labels are reported by category.'))
    $datasetRows.Add([object[]]@('OCSVM fitting','Seeded subset of model-fit rows', '10,000', 'RBF kernel; same fixed subset for all thresholds.'))
    Add-Table @('Stage','Dataset / source','Rows','Role and notes') $datasetRows 8
    Add-Paragraph ('Evaluation totals: validation positives = {0}; validation negatives = {1}; zero-day attacks = {2}. The validation class distribution is attack-heavy ({3} attacks and {4} benign samples), so accuracy and precision should be interpreted for this specific split.' -f (Fmt-Count $positiveCount), (Fmt-Count $negativeCount), (Fmt-Count $zeroN), (Fmt-Pct ($positiveCount / $validationN)), (Fmt-Pct ($negativeCount / $validationN)))

    Add-Heading '4 Preprocessing' 1
    Add-Paragraph 'The experiment consumes the existing prepared/scaled CSV values. Repository documentation identifies the prepared feature values as StandardScaler output. No scaler artifact or raw-to-scaled transformation script was present, so this experiment did not refit a scaler and cannot verify the scaler''s original fitting population. For scoring, positive and negative infinity were replaced by missing values, missing feature values were filled with zero, and arrays were cast to float32. Labels use numeric binary_label; where missing, label equal to BenignTraffic maps to 0 and every other label maps to 1.'
    Add-Paragraph 'The benign-only source has 878,328 rows. A seeded uniform 600,000-row pool was selected by random priorities and split into 500,000 fitting rows and 100,000 reference rows. These row sets are disjoint. The remaining 278,328 source rows were not selected into this experiment sample.'

    Add-Heading '5 Feature engineering and feature order' 1
    Add-Paragraph 'The model input is the repository''s saved 43-column list. The list was applied verbatim and in order for fitting, calibration, validation, zero-day scoring, and inference. The inspected repository contains the final prepared columns but not the raw feature-engineering pipeline, raw input schema, or scaler fit artifact; therefore raw feature count and removed raw columns cannot be reconstructed here. The prepared names include indicator and ratio features already present in the data.'
    Add-TableCaption 'Table 2. Ordered model input features.'
    $featureRows = [System.Collections.Generic.List[object[]]]::new()
    for ($i = 0; $i -lt 22; $i++) {
        $leftN = $i + 1; $rightN = $i + 23
        $left = if ($leftN -le $featureList.Count) { $featureList[$leftN - 1] } else { '' }
        $right = if ($rightN -le $featureList.Count) { $featureList[$rightN - 1] } else { '' }
        $featureRows.Add([object[]]@([string]$leftN, $left, [string]$rightN, $right))
    }
    Add-Table @('Index','Feature name','Index','Feature name') $featureRows 8

    Add-Heading '6 Models' 1
    Add-Paragraph 'The existing six-model script defines the architectures and scoring functions. AE, DAE, and CAE use mean squared reconstruction error. VAE is scored deterministically by decoding the posterior mean (mu) and taking mean squared reconstruction error. Deep SVDD uses squared distance from its fixed latent center. OCSVM anomaly score is the negative decision_function output, so larger values indicate more anomalous observations.'
    Add-Paragraph 'The separately saved Deep_SVDD_Tuned.pt checkpoint (43-128-64-16) was inspected but not reused: the requested six-model training script defines the Deep SVDD architecture used in this comparison as 43-64-32-16. Existing model and XAI outputs were left intact; new checkpoints are saved under this experiment directory.'

    Add-Heading '7 Training configuration' 1
    Add-Paragraph 'Neural models use Adam, learning rate 0.001, batch size 2,048, eight epochs, and no early stopping. The VAE objective is reconstruction MSE plus 0.01 times KL divergence; scoring uses the mean latent vector. DAE adds Gaussian noise with standard deviation 0.05 to the input and reconstructs the clean input. CAE uses a contractive penalty weight of 1e-4. Deep SVDD uses a fixed center computed from the untrained encoder''s mean output, clamps near-zero center coordinates to magnitude 1e-3, and trains with squared center distance and weight decay 1e-6. OCSVM uses an RBF kernel, gamma=scale, nu=0.05, 1,024 MB cache, tolerance 1e-3, shrinking enabled, and random state 42.'
    Start-Landscape
    Add-TableCaption 'Table 3. Model architecture and training configuration.'
    $cfgRows = [System.Collections.Generic.List[object[]]]::new()
    $descriptions = @{
        'AE'=@('43-64-32-16-32-64-43','ReLU hidden layers; linear output','Mean squared reconstruction error','Adam; 0.001','8 epochs; batch 2,048; no early stop')
        'VAE'=@('43-64-32; mu/logvar 16; 16-32-64-43','ReLU hidden layers; linear output','MSE + 0.01 KL; score decode(mu) MSE','Adam; 0.001','8 epochs; batch 2,048; no early stop')
        'DAE'=@('43-64-32-16-32-64-43','ReLU hidden layers; linear output','Clean-target MSE; Gaussian noise std 0.05','Adam; 0.001','8 epochs; batch 2,048; no early stop')
        'CAE'=@('43-64(sigmoid)-16-64(ReLU)-43','Sigmoid encoder; ReLU decoder hidden; linear output','MSE + 1e-4 contractive penalty','Adam; 0.001','8 epochs; batch 2,048; no early stop')
        'Deep SVDD'=@('43-64-32-16; bias-free encoder','ReLU encoder','Squared distance to fixed center; weight decay 1e-6','Adam; 0.001','8 epochs; batch 2,048; no early stop')
        'OCSVM'=@('RBF OneClassSVM; gamma=scale; nu=0.05','Kernel detector; score = -decision_function','No neural loss; 517 support vectors','LibSVM; no gradient optimizer','10,000 benign fit rows; 1,024 MB cache')
    }
    foreach ($m in $modelNames) {
        $mr = $models | Where-Object Model -eq $m | Select-Object -First 1
        $desc = $descriptions[$m]
        $cfgRows.Add([object[]]@($m, $desc[0], $desc[1], $desc[2], $desc[3], $desc[4], (Fmt-Num $mr.Training_Time_Seconds 2), '42 / CPU'))
    }
    Add-Table @('Model','Architecture','Activations / scoring','Objective / regularization','Optimizer / learning rate','Epochs / batch or fit rows','Fit time (s)','Seed / device') $cfgRows 7
    End-Landscape
    Add-Paragraph 'Observed fit times are the model fitting stage, excluding the separate full-file scoring passes. The VAE epoch losses were 0.5229, 0.1922, 0.1345, 16.7317, 0.1179, 0.1105, 0.1046, and 0.0968. The fourth-epoch spike is reported as observed; training still completed and the saved final model was evaluated.'

    Add-Heading '8 Threshold methodology' 1
    Add-Paragraph 'For each fitted model, anomaly scores were computed on the 100,000 disjoint benign reference rows. Thresholds were calculated using NumPy''s default linear percentile method at exactly 90, 93, 95, 97, and 99. A threshold was frozen before evaluation and then applied without change to every validation and zero-day row. A sample is predicted anomalous when its score is greater than or equal to the threshold. Attack labels and attack scores were not used in threshold calibration or model fitting.'
    Add-Paragraph 'All five percentiles are operating points on the same fitted model for each detector. Percentiles do not trigger a new training run. This design avoids the original in-sample training-score calibration used in the project script.'
    Add-TableCaption 'Table 4. Thresholds calculated from benign reference scores only.'
    $thresholdRows = [System.Collections.Generic.List[object[]]]::new()
    foreach ($m in $modelNames) {
        $tr = $thresholds | Where-Object Model -eq $m | Select-Object -First 1
        $thresholdRows.Add([object[]]@($m,(Fmt-Num $tr.Threshold_90 6),(Fmt-Num $tr.Threshold_93 6),(Fmt-Num $tr.Threshold_95 6),(Fmt-Num $tr.Threshold_97 6),(Fmt-Num $tr.Threshold_99 6)))
    }
    Add-Table @('Model','90th','93rd','95th','97th','99th') $thresholdRows 8

    Add-Heading '9 Evaluation metrics' 1
    Add-Paragraph 'For validation, the positive class is attack and the negative class is benign. Accuracy=(TP+TN)/N; precision=TP/(TP+FP); recall and TPR=TP/(TP+FN); F1=2*precision*recall/(precision+recall); FPR=FP/(FP+TN); and specificity=TN/(TN+FP). When a metric denominator is zero, precision, recall, F1, and accuracy are defined as 0; FPR and specificity are undefined if there are no negative evaluation samples. The validation split contains benign rows, so FPR and specificity are defined for the reported results. Zero-day results report attack detection rate, detected, missed, and category sample count.'
    Add-Paragraph 'All files were streamed in 50,000-row chunks. Scores for all five thresholds were accumulated during one pass per model and dataset. The 30 confusion matrices satisfy TP+FN=positive total, TN+FP=negative total, and TP+TN+FP+FN=evaluation total. Precision, recall, F1, FPR, specificity, and zero-day category sums were recomputed and checked.'

    Add-Heading '10 Experimental results' 1
    Add-Paragraph ('The completed evaluation contains {0} validation observations ({1} attacks and {2} benign samples) and {3} zero-day attacks. The best observed validation F1 over all model-percentile points was {4} for {5} at {6}%. The highest zero-day detection was {7} for {8} at {9}%. The lowest validation FPR was {10} for {11} at {12}%.' -f (Fmt-Count $validationN), (Fmt-Count $positiveCount), (Fmt-Count $negativeCount), (Fmt-Count $zeroN), (Fmt-Num $bestF1.F1), $bestF1.Model, $bestF1.Percentile, (Fmt-Pct $bestZD.Zero_Day_Detection_Rate), $bestZD.Model, $bestZD.Percentile, (Fmt-Pct $bestFpr.FPR), $bestFpr.Model, $bestFpr.Percentile)
    Start-Landscape
    Add-TableCaption 'Table 5. Master validation and zero-day metrics for all 30 model-percentile combinations.'
    $masterRows = [System.Collections.Generic.List[object[]]]::new()
    foreach ($r in ($master | Sort-Object { [array]::IndexOf($modelNames, $_.Model) }, { [int]$_.Percentile })) {
        $masterRows.Add([object[]]@($r.Model,[string]$r.Percentile,(Fmt-Num $r.Threshold 5),(Fmt-Pct $r.Accuracy),(Fmt-Pct $r.Precision),(Fmt-Pct $r.Recall),(Fmt-Pct $r.F1),(Fmt-Pct $r.FPR),(Fmt-Pct $r.Specificity),(Fmt-Pct $r.TPR),(Fmt-Pct $r.Zero_Day_Detection_Rate),(Fmt-Count $r.TP),(Fmt-Count $r.TN),(Fmt-Count $r.FP),(Fmt-Count $r.FN)))
    }
    Add-Table @('Model','Pctl','Threshold','Accuracy','Precision','Recall','F1','FPR','Specificity','TPR','Zero-day','TP','TN','FP','FN') $masterRows 7
    End-Landscape

    Add-Heading '11 Percentile comparison' 1
    Add-Paragraph ('Across the six models, median validation FPR decreased from {0} at 90% to {1} at 99%. Median recall changed from {2} to {3}; median F1 from {4} to {5}; median precision from {6} to {7}; and median zero-day detection from {8} to {9}. These are descriptive medians across the six evaluated models at each percentile.' -f (Fmt-Pct $medians[90].FPR),(Fmt-Pct $medians[99].FPR),(Fmt-Pct $medians[90].Recall),(Fmt-Pct $medians[99].Recall),(Fmt-Pct $medians[90].F1),(Fmt-Pct $medians[99].F1),(Fmt-Pct $medians[90].Precision),(Fmt-Pct $medians[99].Precision),(Fmt-Pct $medians[90].ZD),(Fmt-Pct $medians[99].ZD))
    Add-Paragraph ('The measured median pattern shows fewer false positives at higher percentiles, alongside lower recall, F1, and zero-day detection. Median precision rises slightly because the number of false positives falls while true positives remain large in this attack-heavy validation split. Individual models do not share identical changes, so the model-level curves and full table should be used for model-specific claims.')

    Add-Heading '12 Zero-day evaluation' 1
    Add-Paragraph ('The zero-day file contains {0} attack rows across the following nine categories: {1}. No benign rows were used to calculate zero-day detection rate; its denominator is the number of attack samples. At the 95th percentile, Deep SVDD detected {2} of {3} zero-day attacks ({4}) and missed {5}.' -f (Fmt-Count $zeroN), (($categories | Where-Object { [int]$_.Percentile -eq 95 } | Select-Object -ExpandProperty Attack_Category -Unique | Sort-Object) -join ', '), (Fmt-Count $best95ZD.Zero_Day_Detected), (Fmt-Count $best95ZD.Zero_Day_Samples), (Fmt-Pct $best95ZD.Zero_Day_Detection_Rate), (Fmt-Count $best95ZD.Zero_Day_Missed))
    Add-Paragraph 'Category rates are reported for every model and percentile. Because the zero-day file is attack-only, zero-day FPR and specificity are not defined; the full validation file supplies the benign examples used for those metrics.'
    Start-Landscape
    Add-TableCaption 'Table 6. Per-category zero-day samples, detections, misses, and detection rates (all models and percentiles).'
    $categoryRows = [System.Collections.Generic.List[object[]]]::new()
    foreach ($r in ($categories | Sort-Object { [int]$_.Percentile }, Attack_Category, { [array]::IndexOf($modelNames, $_.Model) })) {
        $categoryRows.Add([object[]]@($r.Model,[string]$r.Percentile,$r.Attack_Category,(Fmt-Count $r.Samples),(Fmt-Count $r.Detected),(Fmt-Count $r.Missed),(Fmt-Pct $r.Detection_Rate)))
    }
    Add-Table @('Model','Pctl','Attack category','Samples','Detected','Missed','Detection rate') $categoryRows 7
    End-Landscape

    Add-Heading '13 Visualizations' 1
    Add-Paragraph 'The figures use identical percentile positions and consistent model labels. Performance rates are expressed as percentages; threshold plots retain each model''s score scale. Confusion matrices use the 95th-percentile validation point. The category heatmap uses the 95th percentile, while a second heatmap shows overall zero-day detection over all five percentiles.'
    $zd95cats = @($categories | Where-Object { [int]$_.Percentile -eq 95 })
    $topCat = $zd95cats | Sort-Object { [double]$_.Detection_Rate } -Descending | Select-Object -First 1
    $lowCat = $zd95cats | Sort-Object { [double]$_.Detection_Rate } | Select-Object -First 1
    Add-Figure 1 'all_models_percentile_comparison.png' 'Comparative performance of all six models across the five benign-reference percentiles.' ('The panels show accuracy, precision, recall, F1, FPR, specificity, and zero-day detection together. At the primary 95th percentile, {0} has the highest F1 ({1}) and {2} has the highest zero-day detection ({3}).' -f $best95F1.Model,(Fmt-Num $best95F1.F1),$best95ZD.Model,(Fmt-Pct $best95ZD.Zero_Day_Detection_Rate))
    Add-Figure 2 'f1_vs_percentile.png' 'Validation F1 by percentile.' ('The highest observed F1 is {0} for {1} at {2}%; the lowest is {3} for {4} at {5}%.' -f (Fmt-Num $bestF1.F1),$bestF1.Model,$bestF1.Percentile,(Fmt-Num ((Best-Row 'F1' -Ascending).F1)),(Best-Row 'F1' -Ascending).Model,(Best-Row 'F1' -Ascending).Percentile)
    Add-Figure 3 'recall_vs_percentile.png' 'Validation recall and true-positive rate by percentile.' ('The highest recall is {0} for {1} at {2}%. Recall declines as thresholds become stricter in the six-model median, from {3} at 90% to {4} at 99%.' -f (Fmt-Pct $bestRecall.Recall),$bestRecall.Model,$bestRecall.Percentile,(Fmt-Pct $medians[90].Recall),(Fmt-Pct $medians[99].Recall))
    Add-Figure 4 'fpr_vs_percentile.png' 'Validation false-positive rate by percentile.' ('The minimum FPR is {0} for {1} at {2}%. Median FPR falls from {3} at 90% to {4} at 99%.' -f (Fmt-Pct $bestFpr.FPR),$bestFpr.Model,$bestFpr.Percentile,(Fmt-Pct $medians[90].FPR),(Fmt-Pct $medians[99].FPR))
    Add-Figure 5 'zero_day_vs_percentile.png' 'Overall zero-day attack detection rate by percentile.' ('The maximum zero-day detection is {0} for {1} at {2}%. Median detection is {3} at 90% and {4} at 99%.' -f (Fmt-Pct $bestZD.Zero_Day_Detection_Rate),$bestZD.Model,$bestZD.Percentile,(Fmt-Pct $medians[90].ZD),(Fmt-Pct $medians[99].ZD))
    Add-Figure 6 'precision_vs_percentile.png' 'Validation precision by percentile.' ('The highest precision is {0} for {1} at {2}%. The six-model median ranges from {3} at 90% to {4} at 99%.' -f (Fmt-Pct $bestPrecision.Precision),$bestPrecision.Model,$bestPrecision.Percentile,(Fmt-Pct $medians[90].Precision),(Fmt-Pct $medians[99].Precision))
    Add-Figure 7 'accuracy_vs_percentile.png' 'Validation accuracy by percentile.' ('The 90% median accuracy is {0}; at 99% it is {1}. Accuracy is reported for the observed attack-heavy validation class distribution.' -f (Fmt-Pct $medians[90].Accuracy),(Fmt-Pct $medians[99].Accuracy))
    Add-Figure 8 'threshold_vs_percentile.png' 'Anomaly-score threshold by model and percentile.' 'Each curve is the empirical benign-reference score percentile for one fixed fitted model. Score scales differ across models, so compare within-model movement rather than threshold magnitudes between models.'
    Add-Figure 9 'confusion_matrix_comparison.png' 'Six validation confusion matrices at the 95th percentile.' ('The figure uses the same {0} positive and {1} negative validation samples for every model. Deep SVDD records TP={2}, FN={3}, FP={4}, and TN={5} at this operating point.' -f (Fmt-Count $positiveCount),(Fmt-Count $negativeCount),(Fmt-Count $best95F1.TP),(Fmt-Count $best95F1.FN),(Fmt-Count $best95F1.FP),(Fmt-Count $best95F1.TN))
    Add-Figure 10 'zero_day_heatmap.png' 'Category-level zero-day detection rate by model at the 95th percentile.' ('Across model-category cells, the highest detection is {0} for {1} on {2}; the lowest is {3} for {4} on {5}.' -f (Fmt-Pct $topCat.Detection_Rate),$topCat.Model,$topCat.Attack_Category,(Fmt-Pct $lowCat.Detection_Rate),$lowCat.Model,$lowCat.Attack_Category)
    Add-Figure 11 'zero_day_percentile_heatmap.png' 'Overall zero-day detection by model and percentile.' ('Deep SVDD records {0} zero-day detection at 90% and {1} at 99%; the heatmap makes the change across all models visible.' -f (Fmt-Pct (($master | Where-Object {$_.Model -eq 'Deep SVDD' -and [int]$_.Percentile -eq 90}).Zero_Day_Detection_Rate)),(Fmt-Pct (($master | Where-Object {$_.Model -eq 'Deep SVDD' -and [int]$_.Percentile -eq 99}).Zero_Day_Detection_Rate)))

    Add-Heading '14 Model-by-model analysis' 1
    foreach ($m in $modelNames) {
        $rows = @($master | Where-Object Model -eq $m | Sort-Object { [int]$_.Percentile })
        $p95 = $rows | Where-Object { [int]$_.Percentile -eq 95 } | Select-Object -First 1
        $br = $bestPoints | Where-Object Model -eq $m | Select-Object -First 1
        $mr = $models | Where-Object Model -eq $m | Select-Object -First 1
        $bestM = $rows | Sort-Object { [double]$_.F1 } -Descending | Select-Object -First 1
        $bestZ = $rows | Sort-Object { [double]$_.Zero_Day_Detection_Rate } -Descending | Select-Object -First 1
        $lowF = $rows | Sort-Object { [double]$_.FPR } | Select-Object -First 1
        $f1Vals = @($rows | ForEach-Object { [double]$_.F1 }); $zdVals = @($rows | ForEach-Object { [double]$_.Zero_Day_Detection_Rate })
        Add-Heading $modelLabels[$m] 2
        Add-Paragraph ('Architecture/configuration: {0}. Fit used {1} benign rows in {2} seconds; final logged loss was {3}. Seed 42; CPU.' -f $mr.Architecture,(Fmt-Count $mr.Training_Samples),(Fmt-Num $mr.Training_Time_Seconds 2),$(if ($null -ne $mr.Final_Training_Loss -and $mr.Final_Training_Loss -ne '') { Fmt-Num $mr.Final_Training_Loss 6 } else { 'not applicable to OCSVM' }))
        Add-Paragraph ('At 95%: threshold={0}; accuracy={1}; precision={2}; recall/TPR={3}; F1={4}; FPR={5}; specificity={6}; zero-day detection={7}. Confusion counts: TP={8}, TN={9}, FP={10}, FN={11}.' -f (Fmt-Num $p95.Threshold 6),(Fmt-Pct $p95.Accuracy),(Fmt-Pct $p95.Precision),(Fmt-Pct $p95.Recall),(Fmt-Pct $p95.F1),(Fmt-Pct $p95.FPR),(Fmt-Pct $p95.Specificity),(Fmt-Pct $p95.Zero_Day_Detection_Rate),(Fmt-Count $p95.TP),(Fmt-Count $p95.TN),(Fmt-Count $p95.FP),(Fmt-Count $p95.FN))
        Add-Paragraph ('Across percentiles, F1 ranges from {0} to {1}; zero-day detection ranges from {2} to {3}. Highest F1: {4}% ({5}); highest zero-day detection: {6}% ({7}); lowest FPR: {8}% ({9}). The descriptive balanced score selects {10}% for this model; this post-evaluation summary does not alter threshold calibration.' -f (Fmt-Num (($f1Vals | Measure-Object -Minimum).Minimum)),(Fmt-Num (($f1Vals | Measure-Object -Maximum).Maximum)),(Fmt-Pct (($zdVals | Measure-Object -Minimum).Minimum)),(Fmt-Pct (($zdVals | Measure-Object -Maximum).Maximum)), $bestM.Percentile,(Fmt-Num $bestM.F1),(Fmt-Num $bestZ.Percentile 0),(Fmt-Pct $bestZ.Zero_Day_Detection_Rate),(Fmt-Num $lowF.Percentile 0),(Fmt-Pct $lowF.FPR),$br.Best_Balanced_Descriptive_Percentile)
    }

    Add-Heading '15 Cross-model comparison' 1
    Add-Paragraph ('Across all 30 operating points, {0} at {1}% has the highest F1 ({2}); {3} at {4}% has the highest zero-day detection ({5}); {6} at {7}% has the lowest FPR ({8}); {9} at {10}% has the highest recall ({11}); and {12} at {13}% has the highest precision ({14}). These are different objectives and should not be collapsed into a single ranking.' -f $bestF1.Model,$bestF1.Percentile,(Fmt-Num $bestF1.F1),$bestZD.Model,$bestZD.Percentile,(Fmt-Pct $bestZD.Zero_Day_Detection_Rate),$bestFpr.Model,$bestFpr.Percentile,(Fmt-Pct $bestFpr.FPR),$bestRecall.Model,$bestRecall.Percentile,(Fmt-Pct $bestRecall.Recall),$bestPrecision.Model,$bestPrecision.Percentile,(Fmt-Pct $bestPrecision.Precision))
    Add-Paragraph 'At 95%, Deep SVDD is the leading model on validation recall, F1, and zero-day detection among the six models. Its FPR is also the lowest of the six at that percentile (4.9902%). This conclusion is specific to the evaluated dataset and the fixed 95th-percentile threshold rule.'
    $compareRows = [System.Collections.Generic.List[object[]]]::new()
    foreach ($r in $at95) {
        $compareRows.Add([object[]]@($r.Model,(Fmt-Num $r.Threshold 6),(Fmt-Pct $r.Accuracy),(Fmt-Pct $r.Precision),(Fmt-Pct $r.Recall),(Fmt-Pct $r.F1),(Fmt-Pct $r.FPR),(Fmt-Pct $r.Specificity),(Fmt-Pct $r.Zero_Day_Detection_Rate)))
    }
    Add-Paragraph 'The dedicated 95th-percentile comparison table is presented with the best-operating-point tables in Section 17.'

    Add-Heading '16 Threshold trade-offs' 1
    Add-Paragraph ('The median FPR drops by {0} percentage points from 90% to 99%, while median recall drops by {1} points, median F1 by {2} points, and median zero-day detection by {3} points. Median precision rises by {4} points. The thresholds are estimated only from benign reference scores; these outcome changes are measured after applying each frozen threshold to evaluation data.' -f (Fmt-Num (($medians[90].FPR-$medians[99].FPR)*100) 2),(Fmt-Num (($medians[90].Recall-$medians[99].Recall)*100) 2),(Fmt-Num (($medians[90].F1-$medians[99].F1)*100) 2),(Fmt-Num (($medians[90].ZD-$medians[99].ZD)*100) 2),(Fmt-Num (($medians[99].Precision-$medians[90].Precision)*100) 2))
    Add-Paragraph 'At the 95th percentile, the six validation FPRs cluster around 5.0% while recall and zero-day detection vary more widely by model. The threshold percentile therefore does not eliminate model-specific score-distribution differences.'

    Add-Heading '17 Best operating points' 1
    Add-Paragraph ('For a descriptive balance score, the experiment used the geometric mean of validation F1, zero-day detection rate, and validation specificity: cube_root(F1 × zero-day detection × specificity). This score is reported after evaluation and does not determine model fitting or any frozen threshold. The highest such score is {0} for {1} at {2}% (F1={3}, zero-day={4}, FPR={5}). The existing project comparison point remains 95% and is shown separately in Table 8.' -f (Fmt-Num $bestBalanced.Score 4),$bestBalanced.Row.Model,$bestBalanced.Row.Percentile,(Fmt-Num $bestBalanced.Row.F1),(Fmt-Pct $bestBalanced.Row.Zero_Day_Detection_Rate),(Fmt-Pct $bestBalanced.Row.FPR))
    Start-Landscape
    Add-TableCaption 'Table 7. Post-evaluation best points by model and criterion.'
    $operatingRows = [System.Collections.Generic.List[object[]]]::new()
    foreach ($bp in $bestPoints) {
        $overall = $balancedScores | Where-Object { $_.Row.Model -eq $bp.Model } | Sort-Object Score -Descending | Select-Object -First 1
        $balancedRow = $overall.Row
        $operatingRows.Add([object[]]@($bp.Model,("{0}% / {1}" -f $bp.Best_F1_Percentile,(Fmt-Num $bp.Best_F1)),("{0}% / {1}" -f $bp.Best_Zero_Day_Percentile,(Fmt-Pct $bp.Best_Zero_Day_Rate)),("{0}% / {1}" -f $bp.Lowest_FPR_Percentile,(Fmt-Pct $bp.Lowest_FPR)),("{0}% / {1}" -f $bp.Best_Balanced_Descriptive_Percentile,(Fmt-Num $bp.Best_Balanced_Descriptive_Score)),(Fmt-Num $balancedRow.F1),(Fmt-Pct $balancedRow.Zero_Day_Detection_Rate),(Fmt-Pct $balancedRow.FPR)))
    }
    Add-Table @('Model','Best F1 (pctl / F1)','Best zero-day (pctl / rate)','Lowest FPR (pctl / FPR)','Balanced (pctl / score)','F1 at balance','Zero-day at balance','FPR at balance') $operatingRows 7
    Add-TableCaption 'Table 8. Primary cross-model comparison at the 95th percentile.'
    Add-Table @('Model','Threshold','Accuracy','Precision','Recall','F1','FPR','Specificity','Zero-day detection') $compareRows 8
    End-Landscape

    Add-Heading '18 Limitations' 1
    Add-Paragraph 'The repository contains prepared scaled feature matrices but no fitted scaler object or raw transformation code. The StandardScaler fitting population, upstream raw feature count, and full removed-feature list therefore cannot be independently confirmed from these files. No independent scaling was performed.'
    Add-Paragraph 'The 500,000 training and 100,000 reference observations are disjoint rows sampled from the same benign-only source; the split is row-level, not device-, session-, or time-based. Duplicate or near-duplicate flows may exist in the source and were not deduplicated for this run. Only one random seed was evaluated, so no run-to-run confidence interval is available.'
    Add-Paragraph 'OCSVM was fitted on 10,000 benign rows while the other five models used 500,000. This is a documented resource constraint and limits direct comparisons of representation coverage. The full validation file is attack-heavy, and the zero-day file contains attacks only; class prevalence and zero-day detection rates may differ under other traffic mixes.'
    Add-Paragraph 'The VAE showed a large epoch-4 loss spike before returning to a lower final loss. OCSVM scoring generated a feature-name warning in this run because it was fitted on an unnamed array and evaluated on named chunks; feature order was held fixed. The runner now uses NumPy consistently for later reproductions. The reported metrics belong to the completed run before that non-numeric runner cleanup.'
    Add-Paragraph 'Post-evaluation best-point summaries are descriptive and use evaluation labels to compare already-frozen thresholds. They are not a deployable threshold-selection rule and must not be interpreted as a threshold tuned without test exposure. The designated 95th-percentile comparison was specified before reviewing these measured results.'

    Add-Heading '19 Research observations' 1
    Add-Paragraph ('At the 95th percentile, Deep SVDD reaches {0} recall and {1} zero-day detection with {2} FPR. The lowest-FPR operating points occur at 99%: AE has the smallest observed value, {3}, while its recall and zero-day detection at that same point are {4} and {5}. This illustrates the measured trade-off between suppressing benign alerts and retaining attack detections.' -f (Fmt-Pct $best95F1.Recall),(Fmt-Pct $best95ZD.Zero_Day_Detection_Rate),(Fmt-Pct $best95F1.FPR),(Fmt-Pct $bestFpr.FPR),(Fmt-Pct $bestFpr.Recall),(Fmt-Pct $bestFpr.Zero_Day_Detection_Rate))
    Add-Paragraph ('The 95th-percentile category heatmap ranges from {0} ({1}, {2}) to {3} ({4}, {5}). Category-level differences matter: overall zero-day performance is dominated by the three large Mirai categories, while the smaller Web-based categories contribute far fewer samples.' -f (Fmt-Pct $lowCat.Detection_Rate),$lowCat.Model,$lowCat.Attack_Category,(Fmt-Pct $topCat.Detection_Rate),$topCat.Model,$topCat.Attack_Category)

    Add-Heading '20 Conclusion' 1
    Add-Paragraph ('The six existing detectors completed the same benign-only, 43-feature experiment with five thresholds per model, full streaming evaluation, and per-category zero-day results. Deep SVDD provides the strongest 95th-percentile operating point in this run: F1={0}, recall={1}, FPR={2}, and zero-day detection={3}. At a stricter 99th-percentile cutoff, the lowest FPR overall is {4} for AE, while Deep SVDD is the highest composite descriptive balance point. These findings support reporting both the 95th-percentile primary comparison and the full threshold curves rather than naming a universal winner.' -f (Fmt-Num $best95F1.F1),(Fmt-Pct $best95F1.Recall),(Fmt-Pct $best95F1.FPR),(Fmt-Pct $best95ZD.Zero_Day_Detection_Rate),(Fmt-Pct $bestFpr.FPR))

    Add-Heading '21 Reproducibility information' 1
    Add-Paragraph ('Experiment start: {0}; finish: {1}; elapsed wall time: {2} minutes. Seed: 42. Versions: Python {3}, PyTorch {4}, scikit-learn {5}, pandas {6}, NumPy {7}. Hardware: CPU-only execution; {8} logical CPUs reported.' -f $startText,$endText,$durationMin,$config.environment.python,$config.environment.torch,$config.environment.scikit_learn,$config.environment.pandas,$config.environment.numpy,$config.environment.logical_cpus)
    Add-Paragraph 'From the project root in Windows PowerShell, the exact command used to reproduce training and evaluation is:'
    Add-Paragraph "py -3.13 .\model_training\outputs\percentile_comparison_experiment\run_experiment.py" 'Normal'
    Add-Paragraph 'The runner and its input paths are recorded in experiment_config.json. The run writes a new set of model checkpoints, result CSVs, plots, environment information, and log under model_training\outputs\percentile_comparison_experiment. Existing project model and XAI outputs were not overwritten.'

    Add-Heading '22 Appendix' 1
    Add-Heading 'A. Numerical validation' 2
    Add-Paragraph 'The saved sanity_checks.csv contains 30 passing rows. For all model-percentile combinations, TP+FN equals the validation positive count (30,999,419); TN+FP equals the validation negative count (159,654); and TP+TN+FP+FN equals 31,159,073. The six metrics were recomputed from counts and matched. Zero-day detected plus missed equals 2,658,953, and every category detected-plus-missed total matches its category sample count. No metric contains NaN or infinity in the master table, and all rates fall within [0,1].'
    Add-Heading 'B. Existing repository artifacts inspected' 2
    Add-Paragraph 'Data files: train_benign_only.csv, validation_set.csv, zero_day_test_scaled_v2.csv. Model/evaluation scripts: models.py, svdd.py, svdd_eval.py, Iso_forr.py, eval_for_iso_forr.py. Saved model inspected: model_training\outputs\Deep_SVDD_Tuned.pt (not reused or overwritten). Feature list: model_training\outputs\zero_shield_feature_columns.json. XAI: Deep_SVDD_XAI.py and existing XAI outputs. README notes prepared StandardScaler space and no scaler artifact. No root requirements.txt or pyproject.toml was present; runtime versions are recorded above.'
    Add-Heading 'C. Experiment output inventory' 2
    Add-Paragraph 'Core outputs: master_results.csv, threshold_results.csv, zero_day_category_results.csv, model_summary.csv, best_operating_points.csv, experiment_config.json, environment_info.txt, experiment_log.txt, sanity_checks.csv, eleven PNG figures, the five neural/Deep SVDD checkpoints, and the One-Class SVM joblib model. The authoritative written report is this single DOCX.'

    # Header, footer, and page numbering across portrait and landscape sections.
    $firstSection = $doc.Sections.Item(1)
    $firstSection.Headers.Item(1).Range.Text = 'ZeroShield AI  |  Model comparison experiment'
    $firstSection.Headers.Item(1).Range.Font.Name = 'Aptos'
    $firstSection.Headers.Item(1).Range.Font.Size = 8
    $firstSection.Headers.Item(1).Range.Font.Color = 8421504
    $firstSection.Footers.Item(1).PageNumbers.Add(1, $true) | Out-Null
    for ($i = 2; $i -le $doc.Sections.Count; $i++) {
        $doc.Sections.Item($i).Headers.Item(1).LinkToPrevious = $true
        $doc.Sections.Item($i).Footers.Item(1).LinkToPrevious = $true
        $doc.Sections.Item($i).PageSetup.DifferentFirstPageHeaderFooter = $false
    }
    $doc.Fields.Update() | Out-Null
    $doc.SaveAs2($reportPath, 16)
    $pageCount = $doc.ComputeStatistics($wdStatisticPages)
    Write-Output "Saved report: $reportPath"
    Write-Output "Pages: $pageCount; tables: $($doc.Tables.Count); embedded figures: $($doc.InlineShapes.Count)"
    $doc.Close(0)
    $word.Quit()
    [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($doc)
    [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)
}
catch {
    if ($doc) { try { $doc.Close(0) } catch {} }
    if ($word) { try { $word.Quit() } catch {} }
    throw
}
