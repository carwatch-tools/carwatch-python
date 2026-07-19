# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Load current and legacy CARWatch raw CSV logs from files, directories, and ZIP archives.
- Extract sample scans and awakening events from raw app logs.
- Normalize wide CARWatch Study Manager exports.
- Load long- and wide-format laboratory saliva CSV files.
- Merge laboratory values with Study Manager results using swap-aware barcode and tube matching.
- Compute AUC, slope, initial value, maximum value, maximum increase, descriptive features, and grouped standard errors.
- Document and test the complete Study Manager-to-saliva-feature workflow.
