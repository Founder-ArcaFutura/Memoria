# Image Memory Feature Plan

**Overall Progress:** `100%`

## Tasks:

- [x] 🟩 **Step 1: Define image metadata models and validation**
  - [x] 🟩 Introduce an image asset schema capturing file reference, MIME type, size, dimensions, and optional caption within existing Pydantic models.
  - [x] 🟩 Extend memory entry validators to accept image assets while linking them to textual memories when provided.

- [x] 🟩 **Step 2: Prepare storage locations and linking logic**
  - [x] 🟩 Establish a filesystem storage strategy for image files and integrate the storage service to persist uploaded images as files.
  - [x] 🟩 Ensure database metadata references stored file paths, includes file details (size, filename, created_at), and associates images with corresponding memory entries plus an `includes_image` toggle.

- [x] 🟩 **Step 3: Update persistence and retrieval flows**
  - [x] 🟩 Add schema/migration updates so long-term memory records capture image metadata and the `includes_image` indicator without disrupting existing data.
  - [x] 🟩 Modify retrieval and search responses to surface when memories contain images and provide toggled access to linked image metadata and text.

- [x] 🟩 **Step 4: Extend API & SDK handling**
  - [x] 🟩 Adjust API endpoints and SDK methods to detect image payloads within `store_memory` calls, saving files and metadata in one request.
  - [x] 🟩 Update responses and helpers so clients receive clear prompts when memories contain images and optional textual links.

- [x] 🟩 **Step 5: Testing & documentation updates**
  - [x] 🟩 Add regression tests covering storing and recalling memories with images, ensuring toggle-based recall works as intended.
  - [x] 🟩 Refresh developer documentation to outline image upload requirements, metadata fields, and recall behaviour expectations.
