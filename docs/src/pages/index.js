import React from 'react';
import clsx from 'clsx';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import ThemedImage from '@theme/ThemedImage';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Heading from '@theme/Heading';
import CardGrid from '@site/src/components/CardGrid';
import styles from './index.module.css';

const CARDS = [
  {
    emoji: '🚀',
    title: 'Getting Started',
    description: 'Install FFTjax and run your first FFT-based homogenization.',
    to: '/documentation/installation',
  },
  {
    emoji: '📖',
    title: 'Theorie',
    description: 'Background on variational FFT homogenization and the underlying theory.',
    to: '/documentation/theorie',
  },
  {
    emoji: '🧪',
    title: 'Examples',
    description: 'Worked examples demonstrating FFTjax in practice.',
    to: '/documentation/examples',
  },
  {
    emoji: '⚡',
    title: 'Benchmark',
    description: "Check FFTjax's performance on your hardware, CPU or GPU.",
    to: '/documentation/benchmark',
  },
];

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
      </div>
    </header>
  );
}

function WorkflowDiagram() {
  return (
    <div className={styles.workflowImage}>
      <ThemedImage
        alt="FFTjax workflow"
        sources={{
          light: useBaseUrl('/img/fftjax_workflow_light.svg'),
          dark: useBaseUrl('/img/fftjax_workflow_dark.svg'),
        }}
        width="500"
      />
    </div>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description={siteConfig.tagline}>
      <HomepageHeader />
      <main>
        <WorkflowDiagram />
        <CardGrid cards={CARDS} />
      </main>
    </Layout>
  );
}
