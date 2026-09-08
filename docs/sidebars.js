// @ts-check

/**
 * Sidebar for the "documentation" docs plugin instance (routeBasePath
 * 'documentation').
 *
 * @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  docsSidebar: [
    'installation',
    {
      type: 'category',
      label: 'Theorie',
      link: {type: 'doc', id: 'theorie/index'},
      items: [
        'theorie/mechanical',
        'theorie/damage',
      ],
    },
    {
      type: 'category',
      label: 'Examples',
      link: {type: 'doc', id: 'examples/index'},
      items: [
        {
          type: 'category',
          label: 'Mechanical Solvers',
          items: [
            'examples/lin-elastic-strain',
            'examples/lin-elastic-strain-vmap',
            'examples/lin-elastic-mixed-bc',
            'examples/inelastic-j2',
          ],
        },
        {
          type: 'category',
          label: 'Damage & Fracture Solvers',
          items: ['examples/phase-field'],
        },
        {
          type: 'category',
          label: 'Inverse Calibration',
          items: ['examples/inverse-calibration'],
        },
        // Structure-Property is still a placeholder on the Examples page
        // (no doc page yet) -- add a matching category here once one exists.
      ],
    },
    'benchmark',
  ],
};

export default sidebars;
