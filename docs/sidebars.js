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
    {
      type: 'category',
      label: '📖 Theorie',
      link: {type: 'doc', id: 'theorie/index'},
      items: [
        'theorie/mechanical',
        'theorie/damage',
      ],
    },
    {
      type: 'category',
      label: '🧪 Examples',
      link: {type: 'doc', id: 'examples/index'},
      items: [
        {
          type: 'category',
          label: 'Mechanical Solvers',
          items: ['examples/lin-elastic-strain', 'examples/lin-elastic-mixed-bc'],
        },
        {
          type: 'category',
          label: 'Damage & Fracture Solvers',
          items: ['examples/phase-field'],
        },
        // Inverse Calibration and Structure-Property groups are placeholders
        // on the Examples page (no doc pages yet) -- add matching categories
        // here once real pages exist under examples/.
      ],
    },
    'benchmark',
  ],
};

export default sidebars;
