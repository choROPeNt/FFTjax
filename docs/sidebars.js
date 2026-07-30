// @ts-check

/**
 * Sidebar for the "documentation" docs plugin instance (routeBasePath
 * 'documentation'). API docs are a separate plugin instance — see
 * sidebarsApi.js — now that neither instance sits at the site root ('/'),
 * which is what previously made a second instance hit a known Docusaurus
 * route/plugin resolution ambiguity.
 *
 * @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  docsSidebar: [
    'installation',
    'theorie',
    {
      type: 'category',
      label: 'Examples',
      link: {type: 'doc', id: 'examples/index'},
      items: ['examples/lin-elastic-strain'],
    },
    'benchmark',
  ],
};

export default sidebars;
