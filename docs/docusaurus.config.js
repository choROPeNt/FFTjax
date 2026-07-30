// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';
import prismGithubDimmed from './src/theme/prismGithubDimmed.js';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'FFTjax',
  tagline: 'GPU-accelerated, differentiable FFT-based spectral solver framework built on JAX',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://choROPeNt.github.io',
  baseUrl: '/FFTjax/',

  organizationName: 'choROPeNt',
  projectName: 'FFTjax',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',

  // The generated API docs (docs/api/) are plain .md with no JSX/imports needed,
  // and can contain angle-bracket text from Python reprs/defaults (e.g. `<factory>`)
  // that MDX misparses as unclosed JSX tags. 'detect' renders .md files as plain
  // Markdown (safe) while .mdx files (installation.mdx, benchmark.mdx) still get
  // full MDX for their component imports.
  markdown: {
    format: 'detect',
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: 'docs',
          routeBasePath: 'documentation',
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/choROPeNt/FFTjax/tree/main/docs/',
          remarkPlugins: [require('remark-math')],
          rehypePlugins: [require('rehype-katex')],
          // API docs are served by their own plugin instance below (separate
          // sidebar, separate /api path).
          exclude: ['api/**'],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  plugins: [
    [
      '@docusaurus/plugin-content-docs',
      /** @type {import('@docusaurus/plugin-content-docs').Options} */
      ({
        id: 'api',
        path: 'api-docs',
        routeBasePath: 'api',
        sidebarPath: './sidebarsApi.js',
        editUrl: 'https://github.com/choROPeNt/FFTjax/tree/main/docs/',
      }),
    ],
  ],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV',
      crossorigin: 'anonymous',
    },
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'FFTjax',
        logo: {
          alt: 'FFTjax Logo',
          src: 'img/fftjax_logo.svg',
        },
        items: [
          {to: '/documentation', label: 'Documentation', position: 'left'},
          {to: '/api', label: 'API', position: 'left'},
          {
            href: 'https://github.com/choROPeNt/FFTjax',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {label: 'Getting Started', to: '/documentation/installation'},
              {label: 'Theorie', to: '/documentation/theorie'},
              {label: 'Examples', to: '/documentation/examples'},
              {label: 'Benchmark', to: '/documentation/benchmark'},
            ],
          },
          {
            title: 'More',
            items: [
              {label: 'GitHub', href: 'https://github.com/choROPeNt/FFTjax'},
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Christian Düreth. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismGithubDimmed,
      },
    }),
};

export default config;
