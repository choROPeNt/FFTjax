import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

function Card({card}) {
  // useBaseUrl must be called at this component's top level (not inside the
  // parent's .map()), so a site served under a baseUrl (e.g. /FFTjax/) gets
  // it prefixed onto card.image -- a raw <img src="/img/..."> would 404
  // under any non-root baseUrl, same reason BenchmarkChart/the homepage use it.
  const imageSrc = useBaseUrl(card.image);
  return (
    <div className="col col--6 margin-bottom--lg">
      <Link to={card.to} className={clsx('card', styles.card)}>
        {card.image && (
          <div className={styles.cardImageWrapper}>
            <img src={imageSrc} alt="" className={styles.cardImage} />
          </div>
        )}
        <div className="card__header">
          <Heading as="h3">
            {card.emoji} {card.title}
          </Heading>
        </div>
        <div className="card__body">
          <p>{card.description}</p>
        </div>
      </Link>
    </div>
  );
}

/**
 * A grid of clickable navigation cards, e.g. for a section landing page.
 *
 * @param {{cards: {emoji: string, title: string, description: string, to: string, image?: string}[]}} props
 */
export default function CardGrid({cards}) {
  return (
    <div className="container margin-vert--lg">
      <div className="row">
        {cards.map((card) => (
          <Card key={card.title} card={card} />
        ))}
      </div>
    </div>
  );
}
