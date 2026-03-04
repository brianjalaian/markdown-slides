# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a presentation framework combining Markdown slides with Reveal.js (v5.2.0) for creating HTML-based slide decks. Currently used for academic lectures on machine learning topics (Transformers, NLP).

## Build and Development Commands

All build commands run from the `reveal.js/` directory:

```bash
cd reveal.js

# Install dependencies
npm install

# Start development server with live reload
npm start

# Build the framework (compiles JS/CSS to dist/)
npm run build

# Run test suite
npm test
```

To view presentations, open `index.html` in a browser or navigate directly to a slide deck (e.g., `/transformer/index.html`).

## Architecture

### Directory Structure

- **`/reveal.js/`** - Embedded Reveal.js framework (do not modify unless customizing the framework itself)
  - `/js/` - Source JavaScript (controllers, components, utils)
  - `/css/` - SCSS stylesheets and themes
  - `/plugin/` - Plugins (highlight, markdown, math, notes, search, zoom)
  - `/dist/` - Compiled output files
- **`/transformer/`** - Example slide deck for ML lectures
  - `index.html` - HTML wrapper that loads Reveal.js and references slides.md
  - `slides.md` - Markdown source for slides
  - `/figure/` - Images for this deck

### Creating New Slide Decks

1. Create a new directory (e.g., `/my-lecture/`)
2. Copy `transformer/index.html` as a template
3. Create a `slides.md` with your content
4. Add images to a `/figure/` subdirectory

### Slide Markdown Format

- `---` separates horizontal slides
- `--` separates vertical slides within a horizontal slide
- Supports MathJax: inline `$...$` and display `$$...$$`
- Code blocks use highlight.js with Monokai theme

## Configuration

- **`_config.yml`** - Jekyll config for GitHub Pages (kramdown markdown, MathJax enabled)
- **`reveal.js/package.json`** - Dependencies and build scripts
- **`reveal.js/gulpfile.js`** - Gulp tasks for build pipeline

## Deployment

Site deploys to GitHub Pages. Push to `main` branch to trigger deployment.
