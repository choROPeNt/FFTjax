import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

/**
 * A grid of clickable navigation cards, e.g. for a section landing page.
 *
 * @param {{cards: {emoji: string, title: string, description: string, to: string}[]}} props
 */
export default function CardGrid({cards}) {
  return (
    <div className="container margin-vert--lg">
      <div className="row">
        {cards.map((card) => (
          <div key={card.title} className="col col--6 margin-bottom--lg">
            <Link to={card.to} className={clsx('card', styles.card)}>
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
        ))}
      </div>
    </div>
  );
}
