// GitHub's "Dark Dimmed" colorscheme, ported to a prism-react-renderer theme.
//
// Same verified colors as src/_docs_style/gh_dimmed.py (the Pygments version
// built for the earlier Sphinx docs), sourced from @primer/primitives'
// dist/json/colors/dark_dimmed.json -- re-encoded here in prism-react-renderer's
// theme object shape instead of Pygments' Style class shape.

const RED_DANGER = "#e5534b";
const CORAL_KEYWORD = "#f47067";
const ORANGE_VARIABLE = "#f69d50";
const GREEN_TAG = "#8ddb8c";
const BLUE_STRING = "#96d0ff";
const BLUE_CONSTANT = "#6cb6ff";
const PURPLE_ENTITY = "#dcbdfb";
const GRAY_COMMENT = "#768390";
const FG_DEFAULT = "#adbac7";
const BG_DEFAULT = "#22272e";

/** @type {import('prism-react-renderer').PrismTheme} */
const prismGithubDimmed = {
  plain: {
    color: FG_DEFAULT,
    backgroundColor: BG_DEFAULT,
  },
  styles: [
    {
      types: ["comment", "prolog", "doctype", "cdata"],
      style: { color: GRAY_COMMENT, fontStyle: "italic" },
    },
    {
      types: ["punctuation"],
      style: { color: FG_DEFAULT },
    },
    {
      types: ["number", "boolean", "constant", "symbol", "deleted"],
      style: { color: BLUE_CONSTANT },
    },
    {
      types: ["tag", "inserted"],
      style: { color: GREEN_TAG },
    },
    {
      types: ["string", "char", "attr-value", "url"],
      style: { color: BLUE_STRING },
    },
    {
      types: ["builtin"],
      style: { color: BLUE_CONSTANT },
    },
    {
      types: ["selector", "attr-name", "entity"],
      style: { color: ORANGE_VARIABLE },
    },
    {
      types: ["operator", "keyword", "atrule", "important"],
      style: { color: CORAL_KEYWORD },
    },
    {
      types: ["function", "class-name"],
      style: { color: PURPLE_ENTITY, fontWeight: "bold" },
    },
    {
      types: ["regex", "variable"],
      style: { color: ORANGE_VARIABLE },
    },
    {
      types: ["namespace"],
      style: { color: CORAL_KEYWORD, opacity: 0.85 },
    },
    {
      types: ["bold"],
      style: { fontWeight: "bold" },
    },
    {
      types: ["italic"],
      style: { fontStyle: "italic" },
    },
    {
      types: ["deleted-sign"],
      style: { color: RED_DANGER },
    },
  ],
};

module.exports = prismGithubDimmed;
